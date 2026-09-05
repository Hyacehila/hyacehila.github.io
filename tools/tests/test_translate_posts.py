import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / 'translate-posts.py'
spec = importlib.util.spec_from_file_location('translate_posts', MODULE_PATH)
translate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(translate)


class TranslationMetadataTests(unittest.TestCase):
    def test_cached_body_syncs_metadata_in_check_and_force_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source/_posts/example.md'
            target = root / 'source_en/_posts/example.md'
            source.parent.mkdir(parents=True)
            target.parent.mkdir(parents=True)
            source.write_text('---\ntitle: Source\ntitle_en: English\ncategories: [Programming, CS Foundations]\n'
                              'tags: [SQL]\nhidden: true\npermalink: /blog/example/\n---\n\nBody\n', encoding='utf-8')
            front, body = translate.split_source(source.read_text(encoding='utf-8'))
            metadata = translate.english_front_matter(front, translate.sha256(body), 'example')
            metadata.pop('hidden')
            metadata['categories'] = ['Programming', 'Old category']
            metadata['tags'] = ['Old tag']
            original = translate.dump_document(metadata, '<p>Existing translation &amp; formatting.</p>')
            target.write_text(original, encoding='utf-8')
            suffix = original.split('\n---', 1)[1]
            with patch.object(translate, 'ROOT', root), patch.object(translate, 'translation_cache', side_effect=AssertionError('must not load model')):
                for force in [[], ['--force']]:
                    target.write_text(original, encoding='utf-8')
                    with patch('sys.argv', ['translate-posts.py', '--check', *force]), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(translate.main(), 1)
                    self.assertEqual(target.read_text(encoding='utf-8'), original)
                    with patch('sys.argv', ['translate-posts.py', *force]), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(translate.main(), 0)
                    updated = target.read_text(encoding='utf-8')
                    self.assertEqual(updated.split('\n---', 1)[1], suffix)
                    actual, _ = translate.split_source(updated)
                    for key in ['hidden', 'categories', 'tags']:
                        self.assertEqual(actual[key], front[key])
                    with patch('sys.argv', ['translate-posts.py', '--check', *force]), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(translate.main(), 0)

    def test_removed_optional_metadata_is_not_kept_in_translation(self):
        front = {'title': 'Source', 'categories': ['Programming', 'CS Foundations'], 'tags': ['SQL']}
        target = translate.dump_document({**translate.english_front_matter(front, 'hash', 'example'),
                                          'hidden': True, 'custom_en': 'preserved'}, '<p>Body</p>')
        updated = translate.sync_metadata(front, target, 'hash', 'example')
        actual, _ = translate.split_source(updated)
        self.assertNotIn('hidden', actual)
        self.assertEqual(actual['custom_en'], 'preserved')
        self.assertEqual(translate.sync_metadata(front, updated, 'hash', 'example'), updated)

    def test_metadata_sync_preserves_body_bytes_and_line_endings(self):
        front = {'title': 'Source', 'hidden': True}
        for newline in ['\n', '\r\n']:
            original = translate.dump_document(translate.english_front_matter({}, 'hash', 'example'),
                                              '<p>Existing body</p>\n<pre>  spacing\n</pre>').replace('\n', newline)
            updated = translate.sync_metadata(front, original, 'hash', 'example')
            delimiter = newline + '---'
            self.assertEqual(updated.split(delimiter, 1)[1].encode(), original.split(delimiter, 1)[1].encode())

    def test_drafts_are_not_regenerated_as_published_translations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'source/_posts').mkdir(parents=True)
            for language in ['source', 'source_en']:
                draft = root / language / '_drafts/example.md'
                draft.parent.mkdir(parents=True)
                draft.write_text('---\ntitle: Draft\n---\nBody\n', encoding='utf-8')
            original = (root / 'source_en/_drafts/example.md').read_bytes()
            with patch.object(translate, 'ROOT', root), patch.object(translate, 'translation_cache', side_effect=AssertionError('must not translate drafts')):
                for args in [[], ['--check'], ['--force']]:
                    with patch('sys.argv', ['translate-posts.py', *args]), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(translate.main(), 0)
                self.assertEqual(translate.target_for(root / 'source/_drafts/example.md'), root / 'source_en/_drafts/example.md')
            self.assertFalse((root / 'source_en/_posts/example.md').exists())
            self.assertEqual((root / 'source_en/_drafts/example.md').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
