#!/usr/bin/env python3
"""
judge.py — the self-correction layer: score every render, pick the best seed per card.

This is what separates a pipeline from a batch script. The generator is not reliable per
image and never will be: a seed sweep on a frozen prompt (`runs/2026-08-15_seed-probe`)
showed the same prompt producing a correctly presented object on three seeds out of four.
Twenty-five per cent is far too high to ship and far too low to fix by rewriting prompts —
three rounds of wording changes were spent discovering exactly that. The remaining failures
are a *selection* problem, so the pipeline generates several candidates and chooses.

A vision model scores each candidate against the card it was meant to be — the category it
was assigned, the pose, the object, the style — and returns a verdict per axis plus a
concrete prompt fix when it rejects. The best candidate per card is linked into `selected/`,
and cards where nothing passed are listed for a human.

The rubric deliberately mirrors the defects the journal actually recorded, not a generic
"is this a good image" question: a judge asked something vague agrees with everything.

Requires: ANTHROPIC_API_KEY (or OPENAI_API_KEY for an OpenAI-compatible vision endpoint).

Examples:
    python judge.py --set-dir ../runs/2026-08-14_kino-sets-v12/noir --dry-run
    python judge.py --set-dir ../runs/2026-08-14_kino-sets-v12
    python judge.py --set-dir ... --sheet         # contact sheet of the chosen images
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

from set_designer import CATEGORY_DOC, POSE_TEMPLATES, detect_provider, extract_json

# The axes are the defects the journal recorded, in the order they cost the most work.
AXES = {
    "category": "the background matches the assigned category, no more and no less",
    "pose": "the object is presented in the pose the card asks for",
    "identity": "the object is the one named, recognisable by silhouette alone",
    "style": "casual painterly game art, not a photograph and not a product render",
    "framing": "the object is large, centred, whole, and nothing crops it",
    "cleanliness": "no text, no watermark, no second object, no stray props",
}

# Anything here fails the card outright, whatever the scores say.
HARD_REJECTS = (
    "readable text, lettering or numbers used as an object's identity",
    "a weapon, ammunition, a cartridge case, alcohol, a cigarette or an ashtray",
    # A set may nod to a famous film, but the nod is redrawn as our own object. What must
    # never reach the client is the rights holder's own mark, and it is the generator that
    # adds one: asked for a fedora, Flux occasionally stamps a studio-looking emblem on the
    # band. The ideation stage cannot catch that — only something looking at the picture can.
    "a real brand logo, studio emblem, franchise insignia or copied character likeness",
    "more than one subject in the frame",
    "the object cropped by the frame edge",
)

SYSTEM = """You are the quality gate of an automated pipeline that produces artwork for
collectible cards. You are shown one generated image and the specification it was generated
from. You decide whether this image can ship.

Judge only against the specification. Do not reward an image for being pretty if it is not
the card that was asked for, and do not punish it for choices the specification never made.

Be strict about the pose and the category: those are what hold a set of ten cards together,
and they are exactly what the generator gets wrong. Be concrete in the fix you propose —
"improve the composition" is useless, "the dial is tipped away from the camera" is usable.

Reply with valid JSON only, no commentary."""

USER_TMPL = """Card specification:
- object: {object}
- category {category}: {category_doc}
- required pose: {pose} — {pose_doc}
- background: {background}
- card name (Russian, for context only): {name}

Score each axis from 1 to 5, where 3 is "acceptable to ship" and 5 is "exemplary":
{axes}

Reject outright, whatever the scores, if the image contains any of:
{rejects}

Return JSON:
{{
  "scores": {{{score_keys}}},
  "hard_reject": "the reason, or an empty string",
  "accept": true or false,
  "defect": "the single worst problem in one short phrase, empty if none",
  "fix": "one concrete clause to add to or change in the prompt, empty if none"
}}"""


# --------------------------------------------------------------------------- #
# Vision client
# --------------------------------------------------------------------------- #
class VisionLLM:
    """Thin SDK-free vision client: Anthropic, or an OpenAI-compatible endpoint."""

    def __init__(self, provider: str = "", model: str = "", timeout: int = 120,
                 key: str = ""):
        self.timeout = timeout
        self.provider = provider or detect_provider(key) or (
            "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai")
        if self.provider == "anthropic":
            self.key = key or os.environ.get("ANTHROPIC_API_KEY", "")
            self.model = model or os.environ.get("CARDGEN_JUDGE_MODEL", "claude-sonnet-4-5")
            self.url = "https://api.anthropic.com/v1/messages"
        else:
            self.key = key or os.environ.get("OPENAI_API_KEY", "")
            self.model = model or os.environ.get("CARDGEN_JUDGE_MODEL", "gpt-4o")
            self.url = (os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                        + "/chat/completions")
        if not self.key:
            raise RuntimeError("No API key: type one into the interface, or set "
                               "ANTHROPIC_API_KEY / OPENAI_API_KEY.")
        if not self.key.isascii():
            raise RuntimeError("The key contains non-Latin characters — that does not look like a key.")

    def look(self, system: str, user: str, image: Path) -> str:
        b64 = base64.b64encode(image.read_bytes()).decode()
        if self.provider == "anthropic":
            body = {
                "model": self.model, "max_tokens": 1000, "system": system,
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": user},
                ]}],
            }
            headers = {"content-type": "application/json", "x-api-key": self.key,
                       "anthropic-version": "2023-06-01"}
        else:
            body = {
                "model": self.model, "max_tokens": 1000,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "text", "text": user},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ]},
                ],
            }
            headers = {"content-type": "application/json",
                       "authorization": f"Bearer {self.key}"}

        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode())
        if self.provider == "anthropic":
            return "".join(b.get("text", "") for b in data.get("content", []))
        return data["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
# One image
# --------------------------------------------------------------------------- #
def build_user(card: dict) -> str:
    pose = card.get("pose", "upright")
    return USER_TMPL.format(
        object=card["object"],
        category=card["category"],
        category_doc=CATEGORY_DOC[int(card["category"])],
        pose=pose,
        pose_doc=POSE_TEMPLATES.get(pose, ""),
        background=card.get("bg_style") or "per the category template",
        name=card.get("name", ""),
        axes="\n".join(f"  {k} — {v}" for k, v in AXES.items()),
        rejects="\n".join(f"  - {r}" for r in HARD_REJECTS),
        score_keys=", ".join(f'"{k}": 0' for k in AXES),
    )


def verdict(llm: VisionLLM, card: dict, image: Path, retries: int = 2) -> dict:
    """One verdict, with retries: a malformed reply is a transport problem, not a rejection.

    Failing open would quietly accept unjudged images, so a card whose verdict never parses
    is marked for review instead.
    """
    user = build_user(card)
    for attempt in range(retries + 1):
        try:
            data = extract_json(llm.look(SYSTEM, user, image))
            scores = {k: int(data.get("scores", {}).get(k, 0)) for k in AXES}
            hard = str(data.get("hard_reject", "")).strip()
            accept = bool(data.get("accept")) and not hard and min(scores.values()) >= 3
            return {"scores": scores, "total": sum(scores.values()), "accept": accept,
                    "hard_reject": hard, "defect": str(data.get("defect", "")).strip(),
                    "fix": str(data.get("fix", "")).strip()}
        except Exception as e:
            if attempt == retries:
                return {"scores": {k: 0 for k in AXES}, "total": 0, "accept": False,
                        "hard_reject": "", "defect": f"judge unavailable: {e}", "fix": "",
                        "error": True}
            time.sleep(2 * (attempt + 1))


# --------------------------------------------------------------------------- #
# One variant
# --------------------------------------------------------------------------- #
def candidates(vdir: Path, card: dict, source: str = "raw") -> list[Path]:
    """Every candidate for one card, from whichever stage folder is being judged.

    The judge runs on prepared images, not on raw ones: framing and the category-1 background
    are fixed deterministically before anything is scored, so the verdict is about the picture
    that will actually ship. Judging raw output meant rejecting cards for a floor and a shadow
    that the next stage was about to remove.
    """
    return sorted((vdir / source).glob(f"{card['card_id']}__s*.png"))


def judge_variant(llm: VisionLLM | None, cards_json: Path, out_name: str,
                  dry_run: bool, source: str = "raw") -> dict:
    payload = json.loads(cards_json.read_text(encoding="utf-8"))
    vdir = cards_json.parent
    sel_dir = vdir / out_name
    label = f"{vdir.parent.name}/{vdir.name}"

    results, chosen, review = [], 0, []
    first_try_hits = 0
    judged = 0

    for card in payload["cards"]:
        shots = candidates(vdir, card, source)
        if not shots:
            review.append({"card": card["card_id"], "reason": "no renders found"})
            continue
        if dry_run:
            print(f"  {label} {card['card_id']:14} {card['name']:24} "
                  f"candidates: {len(shots)}")
            judged += len(shots)
            continue

        scored = []
        for i, shot in enumerate(shots):
            v = verdict(llm, card, shot)
            v.update({"image": str(shot), "seed_index": i})
            scored.append(v)
            judged += 1
            mark = "ok" if v["accept"] else "no"
            print(f"  {label} {card['card_id']} [{shot.stem.split('__')[-1]}] "
                  f"{mark}, {v['total']}/30" + (f" — {v['defect']}" if v["defect"] else ""))

        passing = [v for v in scored if v["accept"]]
        best = max(scored, key=lambda v: (v["accept"], v["total"]))
        # "First try" means the first seed rendered for this card was already shippable —
        # the metric the brief cares about, because every extra seed is GPU time.
        if scored and scored[0]["accept"]:
            first_try_hits += 1
        if passing:
            chosen += 1
            sel_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best["image"], sel_dir / f"{card['card_id']}.png")
        else:
            review.append({"card": card["card_id"], "name": card["name"],
                           "reason": best["defect"] or best["hard_reject"],
                           "fix": best["fix"], "best_total": best["total"]})
        results.append({"card_id": card["card_id"], "name": card["name"],
                        "category": card["category"], "pose": card.get("pose", ""),
                        "chosen": str(sel_dir / f"{card['card_id']}.png") if passing else "",
                        "candidates": scored})

    n = len(payload["cards"])
    metrics = {
        "cards": n,
        "generations_total": judged,
        "accepted": chosen,
        "first_try_pass_rate": round(first_try_hits / n, 3) if n else 0.0,
        "needs_review": len(review),
    }
    if not dry_run:
        (vdir / "judge_report.json").write_text(
            json.dumps({"metrics": metrics, "review": review, "cards": results},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {label}: accepted {chosen}/{n}, first-seed "
              f"{metrics['first_try_pass_rate']:.0%}, to review {len(review)}")
    return metrics


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-dir", required=True, help="a set, or a tree of sets")
    ap.add_argument("--source", default="raw",
                    help="which stage to judge: raw, or prepared (after framing and background)")
    ap.add_argument("--out-name", default="selected", help="subfolder for the chosen images")
    ap.add_argument("--provider", default="", choices=["", "anthropic", "openai"])
    ap.add_argument("--model", default="", help="judge model, or CARDGEN_JUDGE_MODEL")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--sheet", action="store_true", help="contact sheet of the chosen images")
    cfg = ap.parse_args()

    root = Path(cfg.set_dir)
    found = sorted(root.glob("**/variant_*/cards.json"))
    if not found:
        sys.exit(f"No variant_*/cards.json found under {root}")

    llm = None if cfg.dry_run else VisionLLM(cfg.provider, cfg.model)
    print(f"Variants to judge: {len(found)}")

    totals = {"cards": 0, "generations_total": 0, "accepted": 0, "needs_review": 0}
    rates = []
    for cards_json in found:
        m = judge_variant(llm, cards_json, cfg.out_name, cfg.dry_run, cfg.source)
        for k in totals:
            totals[k] += m[k]
        rates.append(m["first_try_pass_rate"])

    if cfg.dry_run:
        print(f"\nCandidates to score: {totals['generations_total']}")
        return

    totals["first_try_pass_rate"] = round(sum(rates) / len(rates), 3) if rates else 0.0
    (root / "judge_metrics.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTotal: accepted {totals['accepted']}/{totals['cards']}, "
          f"generations {totals['generations_total']}, "
          f"first-seed {totals['first_try_pass_rate']:.0%}, "
          f"to review {totals['needs_review']}")
    print(f"Metrics: {root / 'judge_metrics.json'}")

    if cfg.sheet:
        build_sheet(root, cfg.out_name)


def build_sheet(root: Path, out_name: str) -> None:
    from PIL import Image, ImageDraw
    from comfy_client import load_font

    shots = sorted(root.glob(f"**/{out_name}/*.png"))
    if not shots:
        return
    W, cols, pad = 220, 10, 6
    probe = Image.open(shots[0])
    h = int(probe.height * W / probe.width)
    rows = (len(shots) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (W + pad) + pad, rows * (h + 22) + pad), (24, 24, 27))
    d = ImageDraw.Draw(sheet)
    f = load_font(13)
    for i, p in enumerate(shots):
        r, c = divmod(i, cols)
        x, y = pad + c * (W + pad), pad + r * (h + 22)
        sheet.paste(Image.open(p).convert("RGB").resize((W, h), Image.LANCZOS), (x, y))
        d.text((x, y + h + 3), p.stem, font=f, fill=(170, 170, 180))
    out = root / "selected_contact.png"
    sheet.save(out)
    print(f"Contact sheet of the selection: {out}")


if __name__ == "__main__":
    main()
