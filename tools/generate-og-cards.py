#!/usr/bin/env python3
"""Generate deterministic 1200x630 PNG social cards for every post."""

from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ARGOS_RUNTIME = ROOT / ".codex-artifacts" / "argos-runtime"
if ARGOS_RUNTIME.exists():
    sys.path.insert(0, str(ARGOS_RUNTIME))

import yaml  # type: ignore  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # type: ignore  # noqa: E402

POSTS = ROOT / "source" / "_posts"
OUTPUT = ROOT / "source" / "assets" / "images" / "og"
WIDTH, HEIGHT = 1200, 630


def front_matter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\r?\n([\s\S]*?)\r?\n---", text)
    return yaml.safe_load(match.group(1)) if match else {}


def card_name(data: dict, fallback: str) -> str:
    permalink = str(data.get("permalink") or "").strip("/")
    return (permalink.split("/")[-1] if permalink else fallback) + ".png"


def find_font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, max_lines: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = char
            if len(lines) == max_lines:
                break
        else:
            current = candidate
    if len(lines) < max_lines and current:
        lines.append(current)
    if len(lines) == max_lines and "".join(lines) != text:
        lines[-1] = lines[-1].rstrip("，。、：； ") + "…"
    return lines


def generate(path: Path) -> Path:
    data = front_matter(path)
    title = str(data.get("title") or path.stem)
    categories = data.get("categories") or []
    if isinstance(categories, str):
        categories = [categories]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#edf5f7")
    gradient = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        t = y / max(1, HEIGHT - 1)
        start = (int(238 - 19 * t), int(247 - 26 * t), int(249 - 8 * t))
        end = (start[0] + 8, start[1] + 3, start[2] + 17)
        gradient.line((0, y, WIDTH, y), fill=start)
        # A soft horizontal tint without an expensive per-pixel Python loop.
        if y % 2 == 0:
            gradient.line((WIDTH * 0.72, y, WIDTH, y), fill=end)

    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((56, 54, 1144, 576), radius=32, fill=(255, 255, 255, 224), outline=(127, 174, 194, 92), width=2)
    draw.ellipse((930, -140, 1280, 210), fill=(184, 167, 217, 42))
    draw.ellipse((-110, 430, 240, 780), fill=(127, 174, 194, 46))

    label_font = find_font(30, bold=True)
    title_font = find_font(58, bold=True)
    meta_font = find_font(26)
    draw.text((104, 100), "HYACEHILA · BLOG", font=label_font, fill="#5f8da1")
    title_lines = wrap(draw, title, title_font, 980, 4)
    y = 176
    for line in title_lines:
        draw.text((104, y), line, font=title_font, fill="#25333d")
        y += 78

    category_text = "  ·  ".join(str(item) for item in categories[-2:]) or "Ideas · Notes · Practice"
    draw.text((104, 516), category_text, font=meta_font, fill="#667784")
    draw.rounded_rectangle((1018, 497, 1096, 551), radius=18, fill="#7faec2")
    draw.text((1041, 507), "H", font=label_font, fill="white")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / card_name(data, path.stem)
    image.save(target, format="PNG", optimize=True)
    return target


def main() -> None:
    files = sorted(POSTS.glob("*.md"))
    for index, path in enumerate(files, 1):
        target = generate(path)
        if index % 25 == 0 or index == len(files):
            print(f"[og] {index}/{len(files)} {target.name}")


if __name__ == "__main__":
    main()
