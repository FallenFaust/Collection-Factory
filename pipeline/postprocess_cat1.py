#!/usr/bin/env python3
"""
postprocess_cat1.py — rebuild category-1 backgrounds so the object really floats.

Category 1 in the brief is an object hovering in the air over a flat colour, gradient or
simple pattern. Flux will not do that: it adds a cast shadow under the object every time.
Three prompt rewrites failed, and a controlled probe showed the shadow appears with the LoRA
at 0.8, at 0.5, and with no LoRA at all — it is the base model's prior, not something wording
can reach.

So the background is not negotiated with the generator, it is replaced. The object is cut out
with a segmentation model, and the backdrop is drawn here, in code, from the `bg_style` already
recorded on the card. That removes the shadow and the invented floor by construction, and as a
side effect the object stops being tinted by whatever the model painted behind it.

The background colour is sampled from the generated image rather than parsed from the palette's
English names ("deep teal"): the pixels the mask calls background are clustered and the dominant
one is kept. The redrawn card then matches the rest of the set by construction, with no
colour-name lookup table to drift out of sync. Its partner tone is derived from it rather than
sampled, because a category-1 backdrop must be two tones of one colour — see `dominant_pair`.

Requires: pip install rembg onnxruntime pillow numpy

Examples:
    python postprocess_cat1.py --set-dir ../runs/2026-08-14_kino-sets-v9/noir --dry-run
    python postprocess_cat1.py --set-dir ../runs/2026-08-14_kino-sets-v9/noir
    python postprocess_cat1.py --set-dir ... --sheet     # before/after contact sheet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

DEFAULT_BG = "gradient"


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def cut_out(img: Image.Image, session=None) -> Image.Image:
    """RGBA copy of the image with the background made transparent."""
    try:
        from rembg import remove
    except ImportError:
        sys.exit(
            "rembg is not installed.\n"
            "  pip install rembg onnxruntime\n"
            "The first run downloads the u2net model (~170 MB); after that it is cached."
        )
    return remove(img.convert("RGB"), session=session).convert("RGBA")


def new_session():
    try:
        from rembg import new_session as ns
    except ImportError:
        return None
    # One session reused across the whole run: loading the model per image would dominate
    # the wall clock for a set of ten cards.
    return ns("u2net")


# --------------------------------------------------------------------------- #
# Colour
# --------------------------------------------------------------------------- #
def dominant_pair(orig: Image.Image, alpha: Image.Image
                  ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """The background's base colour plus a second tone of that same colour.

    The brief allows a category-1 background to be one colour with a gradient, or a simple
    pattern — stripes, a repeating motif, rays. A pattern therefore has to read as two tones
    of a single hue, one lighter and one darker, not as two competing colours: a teal field
    with amber stripes fights the object for attention and stops being a backdrop.

    So only one colour is sampled, and its partner is derived from it. Deriving instead of
    sampling also makes the rule unbreakable — if the generator happened to paint a second
    hue behind the object, it cannot leak into the redrawn card.

    The sample is taken from the ORIGINAL image at the pixels the mask calls background.
    Sampling the cut-out instead would return black: rembg zeroes the RGB channels wherever
    it sets alpha to zero, so the "background colour" of a cut-out is always (0, 0, 0).

    Plain k-means over a subsample — no sklearn dependency for what is twenty lines.
    """
    rgb = np.asarray(orig.convert("RGB"))
    a = np.asarray(alpha)
    bg = rgb[a < 10].astype(np.float64)
    if len(bg) < 50:                      # mask ate everything: fall back to the corners
        bg = np.concatenate([rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3)]).astype(np.float64)
    if len(bg) > 20000:
        bg = bg[np.random.default_rng(0).choice(len(bg), 20000, replace=False)]

    # Four clusters, not two. A striped backdrop carries three or more colours, and with k=2
    # the orange and the teal average into a muddy olive that belongs to neither.
    K = 4
    lum = bg @ np.array([0.299, 0.587, 0.114])
    idx = np.argsort(lum)
    c = bg[idx[np.linspace(0, len(bg) - 1, K).astype(int)]].copy()
    lab = np.zeros(len(bg), dtype=int)
    for _ in range(15):
        d = ((bg[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for k in range(K):
            if (lab == k).any():
                c[k] = bg[lab == k].mean(0)

    # Base colour: whatever covers most of the frame. Deliberately not the darkest — the dark
    # cluster is usually the cast shadow this whole script exists to delete.
    sizes = np.array([(lab == k).sum() for k in range(K)])
    order = np.argsort(sizes)[::-1]
    base = tuple(int(x) for x in c[order[0]].round())
    return base, second_tone(base)


def mix(a, b, t: float):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def luminance(c) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def second_tone(base) -> tuple[int, int, int]:
    """The same colour, one step lighter or darker.

    Mixing towards white or black keeps the hue and only moves lightness, which is exactly
    what "two tones of one colour" means. A dark base is lifted, a light base is deepened, so
    the pattern stays visible either way; a mid grey would otherwise get a partner too close
    to itself to see. The step is small on purpose — the backdrop is meant to be quiet.
    """
    if luminance(base) < 128:
        return mix(base, (255, 255, 255), 0.30)
    return mix(base, (0, 0, 0), 0.24)


# --------------------------------------------------------------------------- #
# Backgrounds — one per kind allowed by the brief
# --------------------------------------------------------------------------- #
def draw_background(size, style: str, c1, c2) -> Image.Image:
    w, h = size
    bg = Image.new("RGB", size, c1)
    d = ImageDraw.Draw(bg)

    if style == "solid":
        return bg

    if style == "gradient":
        grad = np.linspace(0, 1, h)[:, None]
        a, b = np.array(c1, float), np.array(c2, float)
        arr = (a[None, None, :] + (b - a)[None, None, :] * grad[..., None])
        return Image.fromarray(arr.astype(np.uint8).repeat(w, axis=1))

    if style == "stripes":
        band = max(24, w // 9)
        for i in range(-h // band - 1, w // band + 2):      # diagonal 45°
            x = i * band * 2
            d.polygon([(x, 0), (x + band, 0), (x + band + h, h), (x + h, h)], fill=c2)
        return bg.filter(ImageFilter.GaussianBlur(1.2))

    if style == "rays":
        cx, cy, r = w / 2, h * 0.46, (w + h)
        bright = c2 if luminance(c2) > luminance(c1) else mix(c1, (255, 255, 255), 0.30)
        for k in range(16):                                  # every other wedge lit
            if k % 2:
                continue
            a0, a1 = k * 22.5 - 6, k * 22.5 + 6
            d.pieslice([cx - r, cy - r, cx + r, cy + r], a0, a1, fill=bright)
        return bg.filter(ImageFilter.GaussianBlur(w / 55))

    if style == "motif":
        step = max(48, w // 7)
        rad = step // 7
        for row, y in enumerate(range(step // 2, h + step, step)):
            off = 0 if row % 2 == 0 else step // 2
            for x in range(step // 2 + off, w + step, step):
                d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=c2)
        return bg.filter(ImageFilter.GaussianBlur(0.8))

    return bg


# --------------------------------------------------------------------------- #
# One card
# --------------------------------------------------------------------------- #
def process(src: Path, dst: Path, style: str, session) -> dict:
    img = Image.open(src).convert("RGB")
    cut = cut_out(img, session)
    c1, c2 = dominant_pair(img, cut.getchannel("A"))
    bg = draw_background(img.size, style, c1, c2)

    # The object keeps its own pixels from the original: rembg's RGB output is zeroed outside
    # the mask, and its edge is crisp to the point of looking pasted, so re-attach a slightly
    # feathered alpha to the untouched image instead.
    alpha = cut.getchannel("A").filter(ImageFilter.GaussianBlur(0.8))
    cut = img.convert("RGBA")
    cut.putalpha(alpha)

    out = bg.convert("RGBA")
    out.alpha_composite(cut)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(dst, quality=95)

    coverage = float((np.asarray(alpha) > 128).mean())
    # numpy scalars survive arithmetic but not json.dump — cast at the boundary
    return {"style": style, "c1": [int(x) for x in c1], "c2": [int(x) for x in c2],
            "object_coverage": round(coverage, 3)}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def variants(root: Path) -> list[Path]:
    found = sorted(root.glob("**/variant_*/cards.json"))
    if not found:
        sys.exit(f"No variant_*/cards.json found under {root}")
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-dir", required=True, help="a set, or a tree of sets")
    ap.add_argument("--source", default="selected",
                    help="where images come from: selected (after the judge) or raw")
    ap.add_argument("--out-name", default="final", help="subfolder for the result")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--sheet", action="store_true", help="build a before/after sheet")
    cfg = ap.parse_args()

    jobs = []
    for cards_json in variants(Path(cfg.set_dir)):
        vdir = cards_json.parent
        payload = json.loads(cards_json.read_text(encoding="utf-8"))
        for card in payload["cards"]:
            if int(card["category"]) != 1:
                continue
            style = card.get("bg_style") or DEFAULT_BG
            # Two shapes, because the source depends on where in the pipeline this runs:
            # raw/<card_id>__s1000.png straight from the generator, or selected/<card_id>.png
            # after the judge has picked between the seeds. Running on `selected` is the
            # normal order — there is no point redrawing backgrounds for images nobody ships.
            src_dir = vdir / cfg.source
            found = sorted(src_dir.glob(f"{card['card_id']}__s*.png")) \
                or sorted(src_dir.glob(f"{card['card_id']}.png"))
            for src in found:
                jobs.append((src, vdir / cfg.out_name / src.name, style, card["name"]))

    print(f"Category-1 cards to process: {len(jobs)}")
    for src, dst, style, name in jobs[:12]:
        print(f"  {name:26} [{style}]  {src.parent.parent.name}/{src.name}")
    if len(jobs) > 12:
        print(f"  ... and {len(jobs) - 12} more")
    if cfg.dry_run:
        return
    if not jobs:
        return

    session = new_session()
    report = []
    for i, (src, dst, style, name) in enumerate(jobs, 1):
        info = process(src, dst, style, session)
        info.update({"card": name, "src": str(src), "dst": str(dst)})
        report.append(info)
        print(f"[{i}/{len(jobs)}] {name} [{style}] "
              f"background {info['c1']}→{info['c2']}, object covers {info['object_coverage']:.0%}")

    out_root = Path(cfg.set_dir)
    (out_root / "postprocess_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg.sheet:
        build_sheet(report, out_root / "postprocess_before_after.png")
    print(f"\nDone. Report: {out_root / 'postprocess_report.json'}")


def build_sheet(report: list[dict], out: Path) -> None:
    from comfy_client import load_font

    W, pad = 300, 10
    rows = report[:8]
    if not rows:
        return
    probe = Image.open(rows[0]["src"])
    h = int(probe.height * W / probe.width)
    sheet = Image.new("RGB", (pad + 2 * (W + pad), 46 + len(rows) * (h + 30)), (24, 24, 27))
    d = ImageDraw.Draw(sheet)
    f, fs = load_font(18), load_font(14)
    d.text((pad, 10), "before", font=f, fill=(220, 220, 230))
    d.text((pad + W + pad, 10), "after", font=f, fill=(220, 220, 230))
    y = 40
    for r in rows:
        d.text((pad, y), f"{r['card']} [{r['style']}]", font=fs, fill=(160, 160, 170))
        y += 18
        for i, key in enumerate(("src", "dst")):
            sheet.paste(Image.open(r[key]).convert("RGB").resize((W, h), Image.LANCZOS),
                        (pad + i * (W + pad), y))
        y += h + 12
    sheet.crop((0, 0, sheet.width, y)).save(out)
    print(f"Before/after sheet: {out}")


if __name__ == "__main__":
    main()
