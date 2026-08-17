#!/usr/bin/env python3
"""
reframe.py — zoom every card until the object actually fills the frame.

The framing clause in the prompt has been rewritten three times. Each rewrite moved the
object a little larger and none of them landed it where a collectible card needs it: the
object has to read as a thumbnail, and a prop sitting at half the frame height does not.
Wording cannot hold a proportion — the generator has no notion of "70 per cent".

Measuring it, on the other hand, is exact. The object is already segmented for the
category-1 background rebuild, so the same mask gives its bounding box for free. From there
the crop is arithmetic: enlarge the box to the target coverage, keep the aspect ratio, clamp
to the image, resample back to the original size.

Cards that already frame well are left alone — the tolerance exists so that a set does not
get re-encoded for a two per cent gain, which would only cost sharpness.

Requires: pip install rembg onnxruntime pillow numpy

Examples:
    python reframe.py --set-dir ../runs/2026-08-14_kino-sets-v12 --dry-run
    python reframe.py --set-dir ../runs/2026-08-14_kino-sets-v12 --source selected
    python reframe.py --set-dir ... --target 0.8 --sheet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from postprocess_cat1 import cut_out, new_session

# Share of the frame's shorter side that the object's longest dimension should span.
# 0.78 leaves a visible margin on every side while still reading as a close-up; at 0.9 the
# silhouette starts touching the edges, and a card whose object is clipped looks like a bug.
DEFAULT_TARGET = 0.78

# Do not touch a card already within this much of the target: resampling costs sharpness and
# buys nothing the eye can see.
TOLERANCE = 0.06


def object_box(alpha: Image.Image, threshold: int = 128) -> tuple[int, int, int, int] | None:
    """Bounding box of the segmented object, or None if the mask found nothing."""
    a = np.asarray(alpha) > threshold
    if not a.any():
        return None
    ys, xs = np.where(a)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def coverage(box, size) -> float:
    """How much of the frame's shorter side the object's longest dimension spans."""
    x0, y0, x1, y1 = box
    return max(x1 - x0, y1 - y0) / min(size)


def crop_box(box, size, target: float) -> tuple[int, int, int, int]:
    """The crop that puts the object at `target` coverage, centred on it.

    Clamped to the image, so an object already near an edge simply gets the largest crop that
    still fits rather than a crop with black margins — and the aspect ratio is preserved, so
    nothing is stretched.
    """
    w, h = size
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    want = max(x1 - x0, y1 - y0) / target          # desired length of the shorter side
    cw, ch = want * (w / min(size)), want * (h / min(size))
    cw, ch = min(cw, w), min(ch, h)

    # Keep the object centred where possible; slide the window inside the image where not.
    left = min(max(cx - cw / 2, 0), w - cw)
    top = min(max(cy - ch / 2, 0), h - ch)
    return (int(round(left)), int(round(top)),
            int(round(left + cw)), int(round(top + ch)))


def reframe(src: Path, dst: Path, target: float, session) -> dict:
    img = Image.open(src).convert("RGB")
    alpha = cut_out(img, session).getchannel("A")
    box = object_box(alpha)
    if box is None:
        return {"skipped": "mask empty", "before": 0.0, "after": 0.0}

    before = coverage(box, img.size)
    if before >= target - TOLERANCE:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
        return {"skipped": "already framed", "before": round(before, 3),
                "after": round(before, 3)}

    cb = crop_box(box, img.size, target)
    out = img.crop(cb).resize(img.size, Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, quality=95)

    # Report what the crop actually achieved, not what was asked for: the clamp above can cut
    # the zoom short for an object sitting near an edge.
    scale = min(img.size) / min(cb[2] - cb[0], cb[3] - cb[1])
    after = min(before * scale, 1.0)
    return {"skipped": "", "before": round(before, 3), "after": round(after, 3),
            "crop": list(cb)}


def jobs_for(root: Path, source: str, out_name: str) -> list[tuple[Path, Path, str]]:
    jobs = []
    for cards_json in sorted(root.glob("**/variant_*/cards.json")):
        vdir = cards_json.parent
        payload = json.loads(cards_json.read_text(encoding="utf-8"))
        for card in payload["cards"]:
            src_dir = vdir / source
            found = (sorted(src_dir.glob(f"{card['card_id']}__s*.png"))
                     or sorted(src_dir.glob(f"{card['card_id']}.png")))
            for src in found:
                jobs.append((src, vdir / out_name / src.name, card["name"]))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-dir", required=True, help="a set, or a tree of sets")
    ap.add_argument("--source", default="selected",
                    help="where images come from: selected (after the judge) or raw")
    ap.add_argument("--out-name", default="framed", help="subfolder for the result")
    ap.add_argument("--target", type=float, default=DEFAULT_TARGET,
                    help=f"share of the frame's shorter side the object should span, default {DEFAULT_TARGET}")
    ap.add_argument("--dry-run", action="store_true", help="measure only, write nothing")
    ap.add_argument("--sheet", action="store_true", help="before/after sheet")
    cfg = ap.parse_args()

    root = Path(cfg.set_dir)
    jobs = jobs_for(root, cfg.source, cfg.out_name)
    if not jobs:
        sys.exit(f"No images found in {cfg.source}/ under {root}")
    print(f"Cards to measure: {len(jobs)}, target {cfg.target:.0%} of the shorter side")

    session = new_session()
    report, grew = [], 0
    for i, (src, dst, name) in enumerate(jobs, 1):
        if cfg.dry_run:
            img = Image.open(src).convert("RGB")
            box = object_box(cut_out(img, session).getchannel("A"))
            before = coverage(box, img.size) if box else 0.0
            verdict = "ok" if before >= cfg.target - TOLERANCE else "small"
            print(f"[{i}/{len(jobs)}] {name:26} {before:.0%} — {verdict}")
            report.append({"card": name, "before": round(before, 3)})
            continue

        info = reframe(src, dst, cfg.target, session)
        info.update({"card": name, "src": str(src), "dst": str(dst)})
        report.append(info)
        if not info["skipped"]:
            grew += 1
        tail = info["skipped"] or f"{info['before']:.0%} → {info['after']:.0%}"
        print(f"[{i}/{len(jobs)}] {name:26} {tail}")

    if cfg.dry_run:
        small = sum(1 for r in report if r["before"] < cfg.target - TOLERANCE)
        print(f"\nBelow target: {small} of {len(report)}")
        return

    (root / "reframe_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nZoomed: {grew} of {len(jobs)}. Report: {root / 'reframe_report.json'}")

    if cfg.sheet:
        build_sheet([r for r in report if not r.get("skipped")],
                    root / "reframe_before_after.png")


def build_sheet(report: list[dict], out: Path) -> None:
    from comfy_client import load_font

    rows = report[:8]
    if not rows:
        return
    W, pad = 300, 10
    probe = Image.open(rows[0]["src"])
    h = int(probe.height * W / probe.width)
    sheet = Image.new("RGB", (pad + 2 * (W + pad), 46 + len(rows) * (h + 30)), (24, 24, 27))
    d = ImageDraw.Draw(sheet)
    f, fs = load_font(18), load_font(14)
    d.text((pad, 10), "before", font=f, fill=(220, 220, 230))
    d.text((pad + W + pad, 10), "after", font=f, fill=(220, 220, 230))
    y = 40
    for r in rows:
        d.text((pad, y), f"{r['card']}  {r['before']:.0%} → {r['after']:.0%}",
               font=fs, fill=(160, 160, 170))
        y += 18
        for i, key in enumerate(("src", "dst")):
            sheet.paste(Image.open(r[key]).convert("RGB").resize((W, h), Image.LANCZOS),
                        (pad + i * (W + pad), y))
        y += h + 12
    sheet.crop((0, 0, sheet.width, y)).save(out)
    print(f"Before/after sheet: {out}")


if __name__ == "__main__":
    main()
