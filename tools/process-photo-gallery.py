#!/usr/bin/env python3
"""Prepare a private-safe, web-optimized photo gallery for this site.

The source directory is treated as read-only. Published files use content-hash
names and intentionally omit source filenames, capture dates, EXIF, XMP, ICC,
GPS data, and embedded thumbnails.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError


SUPPORTED_SUFFIXES = {".heic", ".heif", ".jpg", ".jpeg", ".png"}
DATE_TAGS = (36867, 36868, 306)  # DateTimeOriginal, DateTimeDigitized, DateTime


@dataclass
class PreparedPhoto:
    source: Path
    source_hash: str
    capture_time: datetime | None
    screenshot: bool
    image: Image.Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Read-only directory containing original photos")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("source/assets/images/photos"),
        help="Gallery image output directory inside the repository",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("source/_data/masonry.yml"),
        help="Generated Redefine masonry data file",
    )
    parser.add_argument("--max-long-edge", type=int, default=2200)
    parser.add_argument("--photo-quality", type=int, default=84)
    parser.add_argument("--screenshot-quality", type=int, default=88)
    parser.add_argument("--photo-max-kib", type=int, default=480)
    parser.add_argument("--screenshot-max-kib", type=int, default=700)
    parser.add_argument(
        "--heif-convert",
        type=Path,
        help="Optional path to heif-convert or its Windows .cmd wrapper",
    )
    return parser.parse_args()


def ensure_inside_repo(path: Path, repo_root: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved != repo_root and repo_root not in resolved.parents:
        raise ValueError(f"{label} must stay inside the repository: {resolved}")
    return resolved


def resolve_heif_converter(explicit: Path | None) -> Path | None:
    if explicit:
        candidate = explicit.expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"heif-convert not found: {candidate}")
        return candidate

    found = shutil.which("heif-convert") or shutil.which("heif-convert.exe")
    return Path(found).resolve() if found else None


def run_heif_convert(converter: Path, source: Path, target: Path) -> None:
    args = [str(converter), "--quiet", str(source), str(target)]
    if converter.suffix.lower() in {".cmd", ".bat"}:
        command_line = subprocess.list2cmdline(args)
        completed = subprocess.run(
            ["cmd.exe", "/d", "/s", "/c", command_line],
            capture_output=True,
            text=True,
        )
    else:
        completed = subprocess.run(args, capture_output=True, text=True)

    if completed.returncode != 0 or not target.exists():
        detail = (completed.stderr or completed.stdout or "HEIF conversion failed").strip()
        raise RuntimeError(f"Could not decode {source.name}: {detail}")


def parse_capture_time(exif: Image.Exif) -> datetime | None:
    for tag in DATE_TAGS:
        value = exif.get(tag)
        if not value:
            continue
        text = str(value).strip().replace("\x00", "")
        for pattern in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, pattern)
            except ValueError:
                pass
    return None


def convert_to_srgb(image: Image.Image, icc_profile: bytes | None) -> Image.Image:
    has_alpha = image.mode in {"RGBA", "LA"} or "transparency" in image.info
    target_mode = "RGBA" if has_alpha else "RGB"
    working = image.convert(target_mode)

    if icc_profile:
        try:
            source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
            srgb_profile = ImageCms.createProfile("sRGB")
            working = ImageCms.profileToProfile(
                working,
                source_profile,
                srgb_profile,
                outputMode=target_mode,
            )
        except (ImageCms.PyCMSError, OSError, ValueError):
            # A malformed or unsupported profile should not block publication.
            working = working.convert(target_mode)

    if has_alpha:
        rgba = working.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        working = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        working = working.convert("RGB")

    return working


def decode_source(source: Path, converter: Path | None, temp_dir: Path) -> tuple[Image.Image, datetime | None]:
    decoded_path: Path | None = None
    try:
        opened = Image.open(source)
    except UnidentifiedImageError:
        if source.suffix.lower() not in {".heic", ".heif"}:
            raise
        if converter is None:
            raise RuntimeError(
                f"{source.name} requires heif-convert, but the executable was not found"
            )
        decoded_path = temp_dir / f"{hashlib.sha256(str(source).encode()).hexdigest()[:16]}.png"
        run_heif_convert(converter, source, decoded_path)
        opened = Image.open(decoded_path)

    with opened as original:
        exif = original.getexif()
        capture_time = parse_capture_time(exif)
        icc_profile = original.info.get("icc_profile")
        oriented = ImageOps.exif_transpose(original)
        oriented.load()
        result = convert_to_srgb(oriented, icc_profile)

    if decoded_path and decoded_path.exists():
        decoded_path.unlink()
    return result, capture_time


def resize_for_web(image: Image.Image, max_long_edge: int) -> Image.Image:
    if max(image.size) <= max_long_edge:
        return image.copy()
    resized = image.copy()
    resized.thumbnail((max_long_edge, max_long_edge), Image.Resampling.LANCZOS)
    return resized


def encode_webp(
    image: Image.Image,
    start_quality: int,
    max_kib: int,
    minimum_quality: int,
) -> tuple[bytes, int]:
    target_bytes = max_kib * 1024
    qualities = list(range(start_quality, minimum_quality - 1, -2))
    if qualities[-1] != minimum_quality:
        qualities.append(minimum_quality)

    encoded = b""
    used_quality = start_quality
    for quality in qualities:
        buffer = io.BytesIO()
        image.save(
            buffer,
            "WEBP",
            quality=quality,
            method=6,
            exact=True,
            exif=b"",
            icc_profile=b"",
            xmp=b"",
        )
        encoded = buffer.getvalue()
        used_quality = quality
        if len(encoded) <= target_bytes:
            break
    return encoded, used_quality


def verify_private_output(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.load()
        exif = image.getexif()
        metadata_keys = {str(key).lower() for key in image.info}
        forbidden = metadata_keys.intersection({"exif", "xmp", "icc_profile"})
        if len(exif) or forbidden:
            raise RuntimeError(f"Private metadata remained in {path.name}: {sorted(forbidden)}")
        if image.format != "WEBP" or image.mode not in {"RGB", "RGBA"}:
            raise RuntimeError(f"Unexpected output format for {path.name}")
        return image.size


def yaml_text(entries: list[dict[str, object]]) -> str:
    lines = [
        "# Generated by tools/process-photo-gallery.py.",
        "# Original filenames, capture dates, EXIF, GPS, and device data are intentionally omitted.",
        "",
    ]
    for entry in entries:
        lines.extend(
            [
                f'- image: "{entry["image"]}"',
                f'  width: {entry["width"]}',
                f'  height: {entry["height"]}',
                "  exif: false",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = args.source.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Photo source directory not found: {source_dir}")

    output_dir = ensure_inside_repo(args.output_dir, repo_root, "output directory")
    data_file = ensure_inside_repo(args.data_file, repo_root, "data file")
    converter = resolve_heif_converter(args.heif_convert)

    source_files = sorted(
        (
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: str(path.relative_to(source_dir)).casefold(),
    )
    if not source_files:
        raise RuntimeError(f"No supported photos found in {source_dir}")

    source_total = sum(path.stat().st_size for path in source_files)
    seen_hashes: set[str] = set()
    prepared: list[PreparedPhoto] = []

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    data_file.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="photo-gallery-decode-") as temp_name:
        temp_dir = Path(temp_name)
        for source in source_files:
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            if source_hash in seen_hashes:
                continue
            seen_hashes.add(source_hash)
            image, capture_time = decode_source(source, converter, temp_dir)
            prepared.append(
                PreparedPhoto(
                    source=source,
                    source_hash=source_hash,
                    capture_time=capture_time,
                    screenshot=source.suffix.lower() == ".png",
                    image=image,
                )
            )

    # Photographs are ordered chronologically when metadata is available.
    # PNG screenshots are placed after camera photos, but capture data is never published.
    prepared.sort(
        key=lambda item: (
            item.screenshot,
            item.capture_time is None,
            item.capture_time or datetime.max,
            item.source.name.casefold(),
        )
    )

    staging_dir = Path(
        tempfile.mkdtemp(prefix=".photo-gallery-staging-", dir=output_dir.parent)
    ).resolve()
    if output_dir.parent not in staging_dir.parents:
        raise RuntimeError(f"Unsafe staging directory: {staging_dir}")

    entries: list[dict[str, object]] = []
    output_total = 0
    qualities: list[int] = []
    try:
        for item in prepared:
            web_image = resize_for_web(item.image, args.max_long_edge)
            start_quality = args.screenshot_quality if item.screenshot else args.photo_quality
            max_kib = args.screenshot_max_kib if item.screenshot else args.photo_max_kib
            minimum_quality = 82 if item.screenshot else 76
            encoded, quality = encode_webp(
                web_image,
                start_quality=start_quality,
                max_kib=max_kib,
                minimum_quality=minimum_quality,
            )
            filename = f"photo-{item.source_hash[:12]}.webp"
            staged_path = staging_dir / filename
            staged_path.write_bytes(encoded)
            width, height = verify_private_output(staged_path)
            output_total += len(encoded)
            qualities.append(quality)
            entries.append(
                {
                    "image": f"/assets/images/photos/{filename}",
                    "width": width,
                    "height": height,
                }
            )
            item.image.close()
            web_image.close()

        output_dir.mkdir(parents=True, exist_ok=True)
        for old_file in output_dir.glob("photo-*.webp"):
            if old_file.is_file():
                old_file.unlink()
        for staged_path in staging_dir.glob("photo-*.webp"):
            shutil.move(str(staged_path), output_dir / staged_path.name)
        data_file.write_text(yaml_text(entries), encoding="utf-8", newline="\n")
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    report = {
        "source_files": len(source_files),
        "published_files": len(entries),
        "exact_duplicates_removed": len(source_files) - len(entries),
        "source_bytes": source_total,
        "output_bytes": output_total,
        "reduction_percent": round((1 - output_total / source_total) * 100, 2),
        "max_long_edge": args.max_long_edge,
        "quality_min": min(qualities),
        "quality_max": max(qualities),
        "output_dir": str(output_dir),
        "data_file": str(data_file),
        "privacy": "EXIF, GPS, XMP, ICC profiles, source filenames, and capture dates omitted",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
