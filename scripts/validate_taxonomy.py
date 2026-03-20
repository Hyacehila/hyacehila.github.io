from __future__ import annotations

import pathlib
import sys
from collections import Counter

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
POSTS_DIR = ROOT / "_posts"

ALLOWED_CATEGORIES = (
    "基础模型",
    "训练与对齐",
    "智能体系统",
    "机器学习",
    "统计学",
    "数据科学",
    "随笔与观察",
)

ALLOWED_TAGS = (
    "Pre-Training",
    "Fine-Tuning",
    "Alignment",
    "Reinforcement Learning",
    "Reward Modeling",
    "Reasoning",
    "Multimodality",
    "Model Mechanics",
    "Agents",
    "Tool Use",
    "MCP",
    "Context Engineering",
    "Retrieval",
    "Evaluation",
    "Data Curation",
    "Ensemble Learning",
    "Interpretability",
    "Imbalanced Learning",
    "Scientific ML",
    "Embeddings",
    "Statistical Inference",
    "Linear Models",
    "Graphical Models",
    "Time Series",
    "Dimensionality Reduction",
    "Resampling",
    "Spatial Data",
    "Data Visualization",
    "Society",
    "Methodology",
)

LEGACY_CATEGORIES = {"LLM", "AI", "Research", "随笔"}


def load_front_matter(path: pathlib.Path) -> dict:
    text = path.read_text(encoding="utf-8-sig")
    if not text.startswith("---"):
        raise ValueError("missing front matter")
    try:
        _, front_matter, _ = text.split("---", 2)
    except ValueError as exc:
        raise ValueError("invalid front matter block") from exc
    return yaml.safe_load(front_matter) or {}


def ensure_list(value) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def validate() -> int:
    errors: list[str] = []
    category_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    per_post_tag_counts: Counter[int] = Counter()

    post_paths = sorted(POSTS_DIR.glob("*.md"))
    if not post_paths:
        print("No published posts found.")
        return 1

    for path in post_paths:
        try:
            data = load_front_matter(path)
        except ValueError as exc:
            errors.append(f"{path.name}: {exc}")
            continue

        categories = ensure_list(data.get("categories"))
        tags = ensure_list(data.get("tags"))

        if len(categories) != 1:
            errors.append(f"{path.name}: categories must contain exactly 1 item, found {len(categories)}")
        else:
            category = categories[0]
            if category in LEGACY_CATEGORIES:
                errors.append(f"{path.name}: legacy category still present: {category}")
            elif category not in ALLOWED_CATEGORIES:
                errors.append(f"{path.name}: unsupported category: {category}")
            else:
                category_counts[category] += 1

        if len(tags) < 2 or len(tags) > 4:
            errors.append(f"{path.name}: tags must contain 2-4 items, found {len(tags)}")
        if len(tags) != len(set(tags)):
            errors.append(f"{path.name}: duplicate tags detected")

        for tag in tags:
            if not isinstance(tag, str) or not tag.strip():
                errors.append(f"{path.name}: empty or non-string tag detected")
                continue
            if "," in tag:
                errors.append(f"{path.name}: tag contains a comma: {tag}")
            if tag not in ALLOWED_TAGS:
                errors.append(f"{path.name}: unsupported tag: {tag}")
            else:
                tag_counts[tag] += 1

        per_post_tag_counts[len(tags)] += 1

    if errors:
        print("Taxonomy validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Validated {len(post_paths)} posts.")
    print("Category counts:")
    for category in ALLOWED_CATEGORIES:
        print(f"- {category}: {category_counts.get(category, 0)}")
    print(f"Unique tags in use: {len(tag_counts)} / {len(ALLOWED_TAGS)}")
    print("Per-post tag counts:")
    for tag_count in sorted(per_post_tag_counts):
        print(f"- {tag_count}: {per_post_tag_counts[tag_count]}")

    unused_tags = [tag for tag in ALLOWED_TAGS if tag not in tag_counts]
    print("Unused tags:")
    if unused_tags:
        for tag in unused_tags:
            print(f"- {tag}")
    else:
        print("- None")

    return 0


if __name__ == "__main__":
    sys.exit(validate())
