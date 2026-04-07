from __future__ import annotations

import re
import sys
from pathlib import Path


ALLOWED_CATEGORIES = {
    "基础模型",
    "训练与对齐",
    "智能体系统",
    "机器学习",
    "数据科学",
    "随笔与观察",
}

FRONT_MATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


def parse_list(raw: str) -> list[str]:
    value = raw.strip()
    if not (value.startswith("[") and value.endswith("]")):
        raise ValueError(f"Expected list syntax, got: {raw}")

    inner = value[1:-1].strip()
    if not inner:
        return []

    return [part.strip().strip("'\"") for part in inner.split(",")]


def parse_front_matter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = FRONT_MATTER_RE.match(text)
    if not match:
        raise ValueError("Missing YAML front matter")

    data: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def validate_post(path: Path) -> tuple[list[str], str | None]:
    errors: list[str] = []

    try:
        front_matter = parse_front_matter(path)
    except ValueError as exc:
        return [f"{path.name}: {exc}"], None

    raw_categories = front_matter.get("categories")
    if raw_categories is None:
        errors.append(f"{path.name}: missing categories")
        return errors, None

    try:
        categories = parse_list(raw_categories)
    except ValueError as exc:
        errors.append(f"{path.name}: invalid categories: {exc}")
        return errors, None

    if len(categories) != 1:
        errors.append(f"{path.name}: categories must contain exactly one entry")
    else:
        category = categories[0]
        if category == "统计学":
            errors.append(f"{path.name}: deprecated category '统计学' is not allowed")
        if category not in ALLOWED_CATEGORIES:
            errors.append(f"{path.name}: category '{category}' is not in the allowed taxonomy")

    raw_tags = front_matter.get("tags")
    if raw_tags is None:
        errors.append(f"{path.name}: missing tags")
    else:
        try:
            tags = parse_list(raw_tags)
        except ValueError as exc:
            errors.append(f"{path.name}: invalid tags: {exc}")
        else:
            if not all(tag and "," not in tag for tag in tags):
                errors.append(f"{path.name}: tags must be non-empty strings without commas")

    series = front_matter.get("series", "").strip().strip("'\"") or None
    category_value = categories[0] if len(categories) == 1 else None
    return errors, (f"{series}\t{category_value}" if series and category_value else None)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    posts_dir = repo_root / "_posts"
    errors: list[str] = []
    series_categories: dict[str, set[str]] = {}

    for post_path in sorted(posts_dir.glob("*.md")):
        post_errors, series_entry = validate_post(post_path)
        errors.extend(post_errors)

        if series_entry:
            series_name, category = series_entry.split("\t", 1)
            series_categories.setdefault(series_name, set()).add(category)

    for series_name, categories in sorted(series_categories.items()):
        if len(categories) > 1:
            categories_display = ", ".join(sorted(categories))
            errors.append(
                f"series '{series_name}' spans multiple categories: {categories_display}"
            )

    if errors:
        print("Taxonomy validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        f"Taxonomy validation passed for {len(list(posts_dir.glob('*.md')))} posts "
        f"across {len(series_categories)} series."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
