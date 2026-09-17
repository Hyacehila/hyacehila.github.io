'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { frontMatter, metadataErrors, sourceDirectoryErrors } = require('../../scripts/validate-i18n');

const source = { categories: ['Programming', 'CS Foundations'], tags: ['SQL', 'Databases'] };

test('flow and block YAML arrays compare identically; missing visibility uses defaults', () => {
  const zh = frontMatter('---\ncategories: [Programming, CS Foundations]\ntags: [SQL, Databases]\n---\n');
  const en = frontMatter('---\ncategories:\n- Programming\n- CS Foundations\ntags:\n- SQL\n- Databases\nhidden: false\n---\n');
  assert.deepEqual(metadataErrors(zh, en), []);
});

for (const key of ['categories', 'tags', 'hidden']) {
  test(`rejects ${key} drift in a bilingual pair`, () => {
    const zh = { ...source };
    const en = { ...zh, [key]: Array.isArray(zh[key]) ? [...zh[key]].reverse() : !zh[key] };
    assert.ok(metadataErrors(zh, en).some(error => error.includes(key)));
  });
}

test('quoted booleans and null are rejected even when both languages agree', () => {
  for (const key of ['hidden']) {
    for (const value of ['false', null, 0]) {
      const fm = { ...source, [key]: value };
      assert.equal(metadataErrors(fm, fm).filter(error => error.includes('boolean')).length, 2);
    }
  }
});


test('Chinese drafts need no translation; published posts must have bilingual pairs', () => {
  assert.deepEqual(sourceDirectoryErrors(['post.md'], ['post.md'], ['draft.md', 'notes.md']), []);
  assert.ok(sourceDirectoryErrors([], ['draft.md'], ['draft.md']).some(error => error.includes('Orphan English translation')));
  assert.ok(sourceDirectoryErrors(['post.md'], [], []).some(error => error.includes('English translation is missing')));
});

test('rejects duplicate Chinese draft/post sources', () => {
  const errors = sourceDirectoryErrors(['same.md'], ['same.md'], ['same.md']);
  assert.deepEqual(errors, ['Chinese source exists in both _posts and _drafts: same.md']);
});
