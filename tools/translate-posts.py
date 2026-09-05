#!/usr/bin/env python3
"""Build committed English source files from the Chinese Hexo source.

The local Argos model translates text nodes after Markdown is rendered to
HTML, so code, URLs, formulas, and element structure remain intact. Generated
files carry a source hash and are skipped when unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARGOS_RUNTIME = ROOT / ".codex-artifacts" / "argos-runtime"
ARGOS_PACKAGES = ROOT / ".codex-artifacts" / "argos-packages"
if ARGOS_RUNTIME.exists():
    sys.path.insert(0, str(ARGOS_RUNTIME))
os.environ.setdefault("ARGOS_PACKAGES_DIR", str(ARGOS_PACKAGES))
os.environ.setdefault("ARGOS_DEVICE_TYPE", "cpu")
os.environ.setdefault("ARGOS_COMPUTE_TYPE", "int8")
os.environ.setdefault("ARGOS_CHUNK_TYPE", "MINISBD")
os.environ.setdefault("ARGOS_BEAM_SIZE", "1")
os.environ.setdefault("PYTHONUTF8", "1")

import yaml  # type: ignore  # noqa: E402

CJK_RE = re.compile(r"[\u3400-\u9fff]")
MATH_OR_URL_RE = re.compile(
    r"(\$\$[\s\S]*?\$\$|(?<!\\)\$(?:\\.|[^$\n])+?(?<!\\)\$|https?://[^\s<]+)",
    re.MULTILINE,
)
SKIP_TAGS = {"code", "pre", "script", "style", "svg", "math", "textarea", "kbd", "samp"}
TRANSLATE_ATTRS = {"alt", "title", "aria-label"}


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_source(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    match = re.match(r"^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n?", text)
    if not match:
        return {}, text
    return yaml.safe_load(match.group(1)) or {}, text[match.end() :]


def render_markdown(markdown: str) -> str:
    result = subprocess.run(
        ["node", str(ROOT / "tools" / "render-markdown.js")],
        input=markdown,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        cwd=ROOT,
    )
    return result.stdout


class TranslationCache:
    def __init__(self, path: Path, translator: Any):
        self.path = path
        self.translator = translator
        self.data: dict[str, str] = {}
        self.dirty = 0
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.data = {}

    def translate(self, text: str) -> str:
        if not CJK_RE.search(text):
            return text
        key = sha256(text)
        cached = self.data.get(key)
        if cached is not None:
            return cached
        translated = self.translator.translate(text)
        self.data[key] = translated
        self.dirty += 1
        if self.dirty % 100 == 0:
            self.flush()
        return translated

    def flush(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        self.dirty = 0


class HTMLTranslator(HTMLParser):
    def __init__(self, cache: TranslationCache):
        super().__init__(convert_charrefs=False)
        self.cache = cache
        self.output: list[str] = []
        self.stack: list[str] = []

    @property
    def skipping(self) -> bool:
        return any(tag in SKIP_TAGS for tag in self.stack)

    def translated_attr(self, value: str) -> str:
        return self.translate_text(value)

    def render_start(self, tag: str, attrs: list[tuple[str, str | None]], closed: bool) -> str:
        parts = [f"<{tag}"]
        for key, value in attrs:
            if value is None:
                parts.append(f" {key}")
                continue
            if key.lower() in TRANSLATE_ATTRS:
                value = self.translated_attr(value)
            parts.append(f' {key}="{html.escape(value, quote=True)}"')
        parts.append("/>" if closed else ">")
        return "".join(parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.output.append(self.render_start(tag, attrs, False))
        self.stack.append(tag.lower())

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.output.append(self.render_start(tag, attrs, True))

    def handle_endtag(self, tag: str) -> None:
        self.output.append(f"</{tag}>")
        lowered = tag.lower()
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index] == lowered:
                del self.stack[index:]
                break

    def translate_text(self, data: str) -> str:
        if not CJK_RE.search(data):
            return data
        pieces = MATH_OR_URL_RE.split(data)
        result: list[str] = []
        for piece in pieces:
            if not piece:
                continue
            if MATH_OR_URL_RE.fullmatch(piece):
                result.append(piece)
                continue
            leading = re.match(r"^\s*", piece).group(0)
            trailing = re.search(r"\s*$", piece).group(0)
            core_end = len(piece) - len(trailing) if trailing else len(piece)
            core = piece[len(leading) : core_end]
            if core and len(core) > 900:
                # Argos inference can stall on very large HTML text nodes;
                # translate bounded chunks while preserving surrounding markup.
                chunks = [core[index : index + 800] for index in range(0, len(core), 800)]
                translated_core = "".join(self.cache.translate(chunk) for chunk in chunks)
                result.append(leading + translated_core + trailing)
            else:
                result.append(leading + self.cache.translate(core) + trailing if core else piece)
        return "".join(result)

    def handle_data(self, data: str) -> None:
        self.output.append(data if self.skipping else self.translate_text(data))

    def handle_entityref(self, name: str) -> None:
        self.output.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.output.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        self.output.append(f"<!--{data}-->")

    def handle_decl(self, decl: str) -> None:
        self.output.append(f"<!{decl}>")


def translate_html(value: str, cache: TranslationCache) -> str:
    parser = HTMLTranslator(cache)
    parser.feed(value)
    parser.close()
    return "".join(parser.output)


def rewrite_english_internal_links(value: str) -> str:
    """Keep contextual article links inside the English URL namespace."""
    return re.sub(r'(?P<prefix>href=["\'])/blog/', r'\g<prefix>/en/blog/', value)


SHARED_METADATA = (
    "date", "updated", "categories", "tags", "author", "mathjax", "cover", "banner", "thumbnail",
    "hidden", "published", "comment", "template", "type", "layout",
)


def english_front_matter(source: dict[str, Any], body_hash: str, stem: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    result["title"] = source.get("title_en") or source.get("title")
    result["title_zh"] = source.get("title")
    for key in SHARED_METADATA:
        if key in source:
            result[key] = source[key]
    excerpt = source.get("excerpt_en") or source.get("description_en") or source.get("excerpt")
    if excerpt:
        result["excerpt"] = excerpt
        result["description"] = excerpt
    if source.get("excerpt"):
        result["excerpt_zh"] = source.get("excerpt")
    if source.get("permalink"):
        permalink = str(source["permalink"])
        result["permalink"] = permalink[3:] if permalink.startswith('/en/') else permalink
    result["lang"] = "en"
    result["translation_key"] = stem
    result["translation_status"] = "machine"
    result["translation_source_hash"] = body_hash
    return result


def dump_document(front_matter: dict[str, Any], body: str, notice: bool = True) -> str:
    head = yaml.safe_dump(
        front_matter,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=120,
    ).rstrip()
    notice_text = (
        '<aside class="translation-notice" role="note">'
        "This English version was machine-translated from the Chinese original. "
        "Technical terms may require verification."
        "</aside>\n\n"
    ) if notice else ""
    return f"---\n{head}\n---\n\n{notice_text}{body.strip()}\n"


def cjk_count(value: str) -> int:
    return len(re.findall(r"[\u3400-\u9fff]", value))


def source_files() -> list[Path]:
    # Fixed pages are maintained as a single English-only edition. Only blog
    # posts participate in the bilingual translation contract.
    return sorted((ROOT / "source" / "_posts").glob("*.md"))


def target_for(source: Path) -> Path:
    relative = source.relative_to(ROOT / "source")
    return ROOT / "source_en" / relative


def sync_metadata(source: dict[str, Any], target_text: str, body_hash: str, stem: str) -> str:
    """Refresh generated metadata without touching the existing translated body."""
    current, _ = split_source(target_text)
    expected = english_front_matter(source, body_hash, stem)
    # Remove obsolete source-owned optional fields while preserving English-only fields.
    managed = set(SHARED_METADATA) | set(expected) | {"excerpt", "description", "excerpt_zh", "permalink"}
    merged = {key: value for key, value in current.items() if key not in managed}
    merged.update(expected)
    if merged == current:
        return target_text
    match = re.match(r"^---\r?\n[\s\S]*?\r?\n---(?=\r?\n|$)", target_text)
    if not match:
        raise ValueError("English source has no YAML front matter")
    head = yaml.safe_dump(merged, allow_unicode=True, sort_keys=False, width=120).rstrip()
    newline = "\r\n" if target_text.startswith("---\r\n") else "\n"
    return f"---\n{head}\n---".replace("\n", newline) + target_text[match.end():]


def translation_cache(shard_count: int, shard: int) -> TranslationCache:
    # Metadata-only updates and --check do not need a translation model.
    try:
        import argostranslate.translate as argos_translate
    except ImportError as exc:
        raise SystemExit(
            "Argos Translate is unavailable. Install it into "
            ".codex-artifacts/argos-runtime and install the zh→en package."
        ) from exc
    translation = argos_translate.get_translation_from_codes("zh", "en")
    if translation is None:
        raise SystemExit("The local zh→en Argos package is not installed.")
    cache_name = "translation-cache-zh-en.json" if shard_count == 1 else f"translation-cache-zh-en-{shard}.json"
    return TranslationCache(ROOT / ".codex-artifacts" / cache_name, translation)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()

    if args.shard_count < 1 or not 0 <= args.shard < args.shard_count:
        parser.error("--shard must be between 0 and --shard-count - 1")

    cache = None

    files = source_files()
    if args.shard_count > 1:
        files = [item for index, item in enumerate(files) if index % args.shard_count == args.shard]
    if args.limit:
        files = files[: args.limit]
    changed = 0
    stale: list[str] = []
    try:
        for index, source_path in enumerate(files, 1):
            source_text = source_path.read_text(encoding="utf-8")
            front, markdown = split_source(source_text)
            body_hash = sha256(markdown)
            target = target_for(source_path)
            if target.exists():
                target_text = target.read_bytes().decode("utf-8")
                target_front, target_body = split_source(target_text)
                if target_front.get("translation_status") == "original" and not args.force:
                    continue
                if (
                    target_front.get("translation_source_hash") == body_hash
                    and (not args.force or cjk_count(target_body) <= 200)
                ):
                    refreshed = sync_metadata(front, target_text, body_hash, source_path.stem)
                    if refreshed != target_text:
                        if args.check:
                            stale.append(str(source_path.relative_to(ROOT)))
                        else:
                            target.write_bytes(refreshed.encode("utf-8"))
                            changed += 1
                            print(f"[translate] metadata {source_path.name}", flush=True)
                    continue
            if args.check:
                stale.append(str(source_path.relative_to(ROOT)))
                continue

            if cache is None:
                cache = translation_cache(args.shard_count, args.shard)
            rendered = render_markdown(markdown)
            translated = rewrite_english_internal_links(translate_html(rendered, cache))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                dump_document(english_front_matter(front, body_hash, source_path.stem), translated),
                encoding="utf-8",
            )
            changed += 1
            print(f"[translate] {index}/{len(files)} {source_path.name}", flush=True)
    finally:
        if cache is not None:
            cache.flush()

    if args.check and stale:
        print("[translate] stale or missing English sources:")
        for item in stale:
            print(f"  - {item}")
        return 1
    print(f"[translate] updated {changed}; checked {len(files)} source files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
