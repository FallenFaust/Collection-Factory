#!/usr/bin/env python3
"""
orchestrator.py — theme and wishes in, finished card sets out.

Everything before this file was a stage that had to be run by hand in the right order with
the right paths. That is a toolbox, not a tool: the brief asks for something a producer can
point at a theme and get a set back from. This is that single call.

    design → generate → prepare (frame + category-1 background) → judge → assemble

Each stage already existed and is imported, not reimplemented. What is added here is the
order, the plumbing between them, a progress stream a caller can render, and a cancel switch
that stops a four-hour render between images rather than at the end.

The result of a run is `<out>/<run_id>/<set>/variant_N/final/` — one PNG per card, named
after the object it shows, plus `manifest.json` with the run's metrics and the Russian card
names that go with each file.

Examples:
    python orchestrator.py --theme "Кино-коллекция: нуар" --wishes "больше латуни и дождя"
    python orchestrator.py --theme "..." --variants 5 --seeds-per-card 2
    python orchestrator.py --theme "..." --offline --no-judge     # smoke test, no API, no GPU
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import set_designer
from batch_generate import DEFAULT_LORA, DEFAULT_WEIGHT, render_variant
from comfy_client import Comfy, build_flux_graph, resolve_models

STAGES = ["design", "review", "generate", "reframe", "judge", "retry", "assemble"]

# File names come from the card's English object description, not from its Russian card name.
# The name is product text and stays Russian in `cards.json` and in the manifest, but a file
# name travels: into git, into an archive, into whatever pipeline picks these up next, and a
# Cyrillic file name survives none of that reliably. "03_cat2_rotary_desk_telephone.png" says
# what the picture is, and which category it belongs to, on any machine.
KEEP = re.compile(r"[^a-z0-9]+")
ARTICLES = ("a ", "an ", "the ")


def safe_name(card: dict) -> str:
    """`03_cat2_rotary_desk_telephone.png` — position, category, object. ASCII only.

    The order is what makes the name useful. Position first, so sorting by name gives the set
    in its own order; then the category, because a folder of ten files is where a producer
    checks that all four categories are present, and counting them by eye should not require
    opening the manifest; then the object, so the file says what it shows.
    """
    obj = str(card.get("object", "")).strip().lower()
    for article in ARTICLES:                     # "a black rotary telephone" -> "black rotary…"
        if obj.startswith(article):
            obj = obj[len(article):]
            break
    slug = KEEP.sub("_", obj.replace("'", "").replace("’", "")).strip("_")[:60].strip("_")
    return f"{int(card['n']):02d}_cat{int(card['category'])}_{slug or card['card_id']}.png"


class Run:
    """State of one production run, shared with whatever is watching it."""

    def __init__(self, theme: str, wishes: str, variants: int, seeds: int):
        self.theme, self.wishes = theme, wishes
        self.variants, self.seeds = variants, seeds
        self.stage = "design"
        self.log: list[str] = []
        self.done, self.total = 0, 0
        self.started = time.time()
        self.finished = False
        self.error = ""
        self.stopping = False
        self.result: dict = {}

    def say(self, message: str) -> None:
        self.log.append(message)
        print(message, flush=True)

    def enter(self, stage: str, total: int = 0) -> None:
        self.stage, self.done, self.total = stage, 0, total

    def as_dict(self) -> dict:
        return {
            "stage": self.stage, "stage_index": STAGES.index(self.stage) if self.stage in STAGES else -1,
            "stages": STAGES, "done": self.done, "total": self.total,
            "elapsed_sec": round(time.time() - self.started),
            "finished": self.finished, "error": self.error,
            "stopping": self.stopping, "log": self.log[-200:], "result": self.result,
        }


def generation_cfg(url: str, lora: str, weight: float, seeds: int, base_seed: int,
                   run: Run) -> SimpleNamespace:
    """The argparse namespace `render_variant` expects, built by hand.

    Listing the defaults here rather than importing an argparse object keeps the generation
    settings frozen by the LoRA bake-off visible in one place — see docs/README_lora_test.md.
    """
    return SimpleNamespace(
        url=url, lora=lora, weight=weight,
        seeds_per_card=seeds, base_seed=base_seed,
        width=832, height=928, steps=20, guidance=3.5,
        thumb=256, timeout=600, out="", csv="",
        dry_run=False, sheets_only=False, template=None, tpl_tokens=set(),
        on_image=lambda i, n, dest: (setattr(run, "done", run.done + 1)),
        should_stop=lambda: run.stopping,
    )


# How many extra attempts a rejected card gets, and how many of them are pure re-seeds.
#
# Re-seeding first is not laziness, it is what the evidence says. A seed sweep on a frozen
# prompt (`runs/2026-08-15_seed-probe`) produced a correctly presented object on three seeds
# out of four: most rejections are noise in the sampler, not a fault in the wording. So the
# first two attempts change nothing but the seed, and only a card that fails twice — which
# means the defect is systematic — gets its prompt rewritten against the reviewer's note.
#
# That order also protects the set. A rewritten prompt drifts away from the nine cards around
# it; a new seed cannot.
MAX_ATTEMPTS = 3
RESEED_ATTEMPTS = 2


def render_one(api, models, cfg, card: dict, prompt: str, seed: int, dest: Path) -> bool:
    """One image for one card, outside the batch runner's per-variant loop."""
    graph = build_flux_graph(
        models, prompt, seed, negative=card.get("negative", ""),
        lora_name=cfg.lora, lora_weight=cfg.weight,
        width=cfg.width, height=cfg.height, steps=cfg.steps, guidance=cfg.guidance,
        filename_prefix=f"cardgen/{card['card_id']}",
    )
    try:
        data = api.render(graph)
    except Exception:
        return False
    if not data:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True


def retry_rejected(api, models, cfg, judge_mod, llm, text_llm, csvs, run,
                   target_coverage: float, session, max_attempts: int = MAX_ATTEMPTS) -> dict:
    """Re-render the cards the judge rejected, judge them again, keep what passes.

    This is the loop the architecture called the core of the pipeline's autonomy, and until now
    it did not exist: the judge produced a verdict and a suggested fix, and a human had to act
    on both. Here the pipeline acts on them itself, up to MAX_ATTEMPTS times per card, and only
    what still fails goes to a person.
    """
    import reframe as reframe_mod
    from postprocess_cat1 import process as redraw

    stats = {"retried": 0, "recovered": 0, "retry_generations": 0, "amended": 0}
    history = []

    for csv_path in csvs:
        vdir = csv_path.parent
        report_path = vdir / "judge_report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rejected = {r["card"] for r in report.get("review", [])}
        if not rejected:
            continue
        payload = json.loads((vdir / "cards.json").read_text(encoding="utf-8"))
        cards = {c["card_id"]: c for c in payload["cards"]}
        worst = {r["card"]: r for r in report.get("review", [])}

        for card_id in sorted(rejected):
            card = cards.get(card_id)
            if not card or run.stopping:
                continue
            stats["retried"] += 1
            prompt = card["prompt"]
            note = worst.get(card_id, {})
            attempts = []

            for attempt in range(1, max_attempts + 1):
                if run.stopping:
                    break
                if attempt > RESEED_ATTEMPTS and text_llm is not None:
                    new_prompt = set_designer.amend_prompt(
                        text_llm, prompt, note.get("reason", ""), note.get("fix", ""))
                    if new_prompt != prompt:
                        prompt, stats["amended"] = new_prompt, stats["amended"] + 1
                        run.say(f"  {card_id}: prompt rewritten against «{note.get('reason','')}»")

                seed = cfg.base_seed + 500 + attempt * 37 + int(card["n"])
                raw = vdir / "raw" / f"{card_id}__r{attempt}.png"
                if not render_one(api, models, cfg, card, prompt, seed, raw):
                    attempts.append({"attempt": attempt, "seed": seed, "error": "render failed"})
                    continue
                stats["retry_generations"] += 1
                run.done += 1

                prepared = vdir / "prepared" / raw.name
                reframe_mod.reframe(raw, prepared, target_coverage, session)
                if int(card["category"]) == 1:
                    redraw(prepared, prepared, card.get("bg_style") or "gradient", session)

                v = judge_mod.verdict(llm, card, prepared)
                attempts.append({"attempt": attempt, "seed": seed, "total": v["total"],
                                 "accept": v["accept"], "defect": v["defect"],
                                 "prompt_changed": prompt != card["prompt"]})
                if v["accept"]:
                    (vdir / "selected").mkdir(parents=True, exist_ok=True)
                    (vdir / "selected" / f"{card_id}.png").write_bytes(prepared.read_bytes())
                    stats["recovered"] += 1
                    run.say(f"  {card_id}: accepted on attempt {attempt}")
                    # The winning prompt is the one that ships, so the run's record matches it.
                    card["prompt"] = prompt
                    break
                note = {"reason": v["defect"], "fix": v["fix"]}
            else:
                run.say(f"  {card_id}: still rejected after {max_attempts} attempts")

            history.append({"card_id": card_id, "name": card.get("name", ""),
                            "attempts": attempts, "final_prompt": prompt})

        # Recompute the review list: whatever passed is no longer waiting for a human.
        still = [r for r in report.get("review", [])
                 if not (vdir / "selected" / f"{r['card']}.png").exists()]
        report["review"] = still
        report["retries"] = history
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (vdir / "cards.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    stats["history"] = history
    return stats


def produce(theme: str, wishes: str = "", variants: int = 3, seeds_per_card: int = 3,
            out: str | Path = "runs", run_id: str = "", offline: bool = False,
            use_judge: bool = True, url: str = "http://127.0.0.1:8188",
            lora: str = DEFAULT_LORA, weight: float = DEFAULT_WEIGHT,
            base_seed: int = 1001, target_coverage: float = 0.78,
            slots: dict | None = None, api_key: str = "", on_designed=None,
            max_attempts: int = MAX_ATTEMPTS, resume: str = "",
            run: Run | None = None) -> dict:
    """One producer request, start to finish. Returns the run manifest."""
    variants = max(1, min(5, int(variants)))
    seeds_per_card = max(1, min(5, int(seeds_per_card)))
    run = run or Run(theme, wishes, variants, seeds_per_card)
    root = Path(out) / (run_id or f"{time.strftime('%Y-%m-%d_%H%M')}_{set_designer.slugify(theme)}")
    if resume:
        candidate = Path(resume)
        root = candidate if candidate.exists() else Path(out) / resume
    root.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. design ------------------------------------------------
    #
    # A run of three sets is four hours of GPU. Losing it to a crash at hour three and
    # starting from a fresh set of invented objects is the difference between an inconvenience
    # and a wasted evening — so a resumed run reuses the design that is already on disk, and
    # the generation stage skips every image that was already rendered.
    designed = sorted(root.glob("*/variant_*/cards.json")) if resume else []
    if designed:
        set_root = designed[0].parent.parent
        plan = json.loads((set_root / "plan.json").read_text(encoding="utf-8"))
        plan["root"] = str(set_root)
        theme = plan.get("theme", theme)
        run.enter("design", len(plan["variants"]))
        run.done = len(plan["variants"])
        rendered = len(list(set_root.glob("variant_*/raw/*.png")))
        run.say(f"Resuming {set_root.name}: {len(designed)} variants already designed, "
                f"{rendered} images already rendered")
    else:
        run.enter("design", variants)
        run.say(f"Theme: {theme}" + (f"; wishes: {wishes}" if wishes else ""))
        plan = set_designer.design_set(theme, wishes, variants, out=root, offline=offline,
                                       slots=slots, api_key=api_key, progress=run.say)
        set_root = Path(plan["root"])
        run.done = len(plan["variants"])

    # ---------------- 2. review ------------------------------------------------
    #
    # The gap between "the model invented ten objects" and "the GPU has spent four hours" is
    # where a producer can still change their mind cheaply. `on_designed` is handed the set
    # directory and may block for as long as it likes; the web interface uses that to show the
    # cards, take edits and wait for a second press. A caller that passes nothing — the CLI,
    # a scheduled run — goes straight through.
    if on_designed is not None and not designed:
        run.enter("review")
        run.say("Set designed. Review the cards, edit anything, then start the generation.")
        on_designed(set_root)
        if run.stopping:
            run.say("Cancelled before generation. Nothing was rendered.")
            run.finished = True
            return {}

    # Re-read after the review: an edit rewrites prompts.csv, and the stale copy would
    # generate exactly the cards the producer just corrected.
    csvs = sorted(set_root.glob("variant_*/prompts.csv"))

    # ---------------- 3. generate ----------------------------------------------
    cards_total = sum(len(json.loads((c.parent / "cards.json").read_text(encoding="utf-8"))["cards"])
                      for c in csvs)
    run.enter("generate", cards_total * seeds_per_card)
    run.say(f"Generating: {len(csvs)} variants × {cards_total // max(len(csvs), 1)} cards "
            f"× {seeds_per_card} seeds = {run.total} images")

    cfg = generation_cfg(url, lora, weight, seeds_per_card, base_seed, run)
    api = Comfy(url, timeout=cfg.timeout)
    api.ping()
    # A LoRA named in the settings but absent in ComfyUI is the single most common setup
    # failure, and it is worth catching before four hours of rendering rather than after.
    available = api.options("LoraLoaderModelOnly", "lora_name")
    if cfg.lora not in available:
        match = [x for x in available if Path(x).name == cfg.lora]
        if len(match) != 1:
            raise RuntimeError(f"ComfyUI cannot see the LoRA \"{cfg.lora}\". Available: "
                               + ", ".join(available[:10]))
        cfg.lora = match[0]
    models = resolve_models(api, "", "", "", "", "")
    run.say(f"Model {models['unet']} [{models['dtype']}], LoRA {Path(cfg.lora).name} @ {weight}")

    for csv_path in csvs:
        if run.stopping:
            break
        render_variant(api, models, csv_path, cfg)

    if run.stopping:
        run.say("Stopped on request. Everything already rendered is on disk.")

    # ---------------- 4. prepare -----------------------------------------------
    #
    # Framing and the category-1 background are deterministic repairs, not polish, so they run
    # on every candidate BEFORE anything is scored. The first version judged raw output and
    # then repaired the winner, which meant category-1 cards were rejected for a floor and a
    # cast shadow that the very next stage removes by construction.
    session = None
    if not run.stopping:
        import reframe as reframe_mod
        from postprocess_cat1 import new_session, process as redraw
        session = new_session()

        prep = []
        for csv_path in csvs:
            vdir = csv_path.parent
            payload = json.loads((vdir / "cards.json").read_text(encoding="utf-8"))
            for card in payload["cards"]:
                for src in sorted((vdir / "raw").glob(f"{card['card_id']}__s*.png")):
                    prep.append((src, vdir / "prepared" / src.name, card))
        run.enter("reframe", len(prep))
        run.say(f"Preparing candidates: {len(prep)} images — framing and category-1 backgrounds")
        grew = 0
        for src, dst, card in prep:
            info = reframe_mod.reframe(src, dst, target_coverage, session)
            grew += 0 if info["skipped"] else 1
            if int(card["category"]) == 1:
                redraw(dst, dst, card.get("bg_style") or "gradient", session)
            run.done += 1
        run.say(f"Zoomed {grew} of {len(prep)}; category-1 backgrounds redrawn")

    # ---------------- 5. judge -------------------------------------------------
    metrics, review = {}, []
    if use_judge and not run.stopping:
        run.enter("judge", len(csvs))
        run.say("The judge scores the prepared candidates and picks one per card")
        import judge as judge_mod
        llm = judge_mod.VisionLLM(key=api_key)
        totals = {"cards": 0, "generations_total": 0, "accepted": 0, "needs_review": 0}
        rates = []
        for csv_path in csvs:
            cards_json = csv_path.parent / "cards.json"
            m = judge_mod.judge_variant(llm, cards_json, "selected", False, source="prepared")
            for k in totals:
                totals[k] += m[k]
            rates.append(m["first_try_pass_rate"])
            report = json.loads((csv_path.parent / "judge_report.json").read_text(encoding="utf-8"))
            review += [{**r, "variant": json.loads(cards_json.read_text(encoding="utf-8"))["variant"]}
                       for r in report["review"]]
            run.done += 1
        totals["first_try_pass_rate"] = round(sum(rates) / len(rates), 3) if rates else 0.0
        metrics = totals
        source = "selected"

        # ---------------- 6. retry ---------------------------------------------
        if review and max_attempts > 1 and not run.stopping:
            run.enter("retry", len(review) * (max_attempts - 1))
            run.say(f"Retrying {len(review)} rejected cards, up to {max_attempts} attempts each")
            text_llm = None if offline else set_designer.LLM(key=api_key)
            rs = retry_rejected(api, models, cfg, judge_mod, llm, text_llm, csvs, run,
                                target_coverage, session, max_attempts)
            metrics.update({k: v for k, v in rs.items() if k != "history"})
            metrics["needs_review"] = totals["needs_review"] - rs["recovered"]
            metrics["accepted"] = totals["accepted"] + rs["recovered"]
            run.say(f"Recovered {rs['recovered']} of {rs['retried']}; "
                    f"{rs['retry_generations']} extra generations, "
                    f"{rs['amended']} prompts rewritten")
            # The review list is rebuilt from the reports the loop just rewrote.
            review = []
            for csv_path in csvs:
                rp = csv_path.parent / "judge_report.json"
                if rp.exists():
                    v = json.loads((csv_path.parent / "cards.json").read_text(encoding="utf-8"))["variant"]
                    review += [{**r, "variant": v}
                               for r in json.loads(rp.read_text(encoding="utf-8"))["review"]]
    else:
        # Without the judge nothing has been checked, and nothing chooses between candidates,
        # so the first one stands in. Stated rather than silent.
        source = "prepared"
        run.say("Judge disabled: the first candidate of each card ships unchecked")

    # ---------------- 7. assemble ----------------------------------------------
    #
    # A card the judge rejected goes to `review/`, never to `final/`. The earlier version fell
    # back through the stage folders until it found any image at all, which quietly delivered
    # rejected cards — worse than having no judge, because the metrics then claim a check that
    # the output does not reflect.
    run.enter("assemble", len(csvs))
    rejected_ids = {r["card"] for r in review}
    out_cards, out_review = [], []
    for csv_path in csvs:
        vdir = csv_path.parent
        payload = json.loads((vdir / "cards.json").read_text(encoding="utf-8"))
        (vdir / "final").mkdir(parents=True, exist_ok=True)
        for card in payload["cards"]:
            ok = card["card_id"] not in rejected_ids
            folder = vdir / ("final" if ok else "review")
            folder.mkdir(parents=True, exist_ok=True)
            for stage in ((source, "prepared") if ok else ("prepared", "raw")):
                found = sorted((vdir / stage).glob(f"{card['card_id']}*.png"))
                if not found:
                    continue
                dest = folder / safe_name(card)
                dest.write_bytes(found[0].read_bytes())
                entry = {"variant": payload["variant"], "n": card["n"], "name": card["name"],
                         "category": card["category"], "rarity": card["rarity"],
                         "pose": card.get("pose", ""), "file": str(dest), "from_stage": stage}
                (out_cards if ok else out_review).append(entry)
                break
        run.done += 1
    if out_review:
        run.say(f"{len(out_review)} cards went to a human — they are in review/, not final/")

    manifest = {
        "theme": theme, "wishes": wishes, "run_dir": str(set_root),
        "variants": len(plan["variants"]), "seeds_per_card": seeds_per_card,
        "slots": plan.get("slots", {}),
        "generation": {"lora": Path(cfg.lora).name, "weight": weight,
                       "steps": 20, "guidance": 3.5, "size": "832x928"},
        "metrics": {**metrics, "wall_time_sec": round(time.time() - run.started),
                    "cards_delivered": len(out_cards)},
        "cards": out_cards,
        "review": out_review,
        "review_reasons": review,
        "plan": plan["variants"],
    }
    (set_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    run.result = manifest
    run.finished = True
    run.say(f"Done: {len(out_cards)} cards in the set"
            + (f", {len(out_review)} need work" if out_review else "")
            + f", {round((time.time()-run.started)/60)} min → {set_root}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theme", required=True, help="set theme")
    ap.add_argument("--wishes", default="", help="the producer's free-form wishes")
    ap.add_argument("--variants", type=int, default=3, help="how many variants of the set, 1-5")
    ap.add_argument("--seeds-per-card", type=int, default=3,
                    help="candidates per card, 1-5; the judge picks one of them")
    ap.add_argument("--out", default="runs", help="where runs are written")
    ap.add_argument("--run-id", default="", help="name of the run folder")
    ap.add_argument("--resume", default="",
                    help="continue an interrupted run: its folder name under --out, or a path. "
                         "The design is reused and rendered images are skipped")
    ap.add_argument("--url", default="http://127.0.0.1:8188", help="ComfyUI address")
    ap.add_argument("--lora", default=DEFAULT_LORA)
    ap.add_argument("--weight", type=float, default=DEFAULT_WEIGHT)
    ap.add_argument("--slots", default="",
                    help="set composition by category, e.g. 2,2,3,3")
    ap.add_argument("--offline", action="store_true", help="no LLM: use the built-in demo set")
    ap.add_argument("--no-judge", action="store_true", help="skip the judge stage")
    ap.add_argument("--attempts", type=int, default=MAX_ATTEMPTS,
                    help=f"attempts per card including the first, default {MAX_ATTEMPTS}; "
                         f"the first {RESEED_ATTEMPTS} only change the seed")
    cfg = ap.parse_args()

    try:
        produce(cfg.theme, cfg.wishes, cfg.variants, cfg.seeds_per_card, cfg.out, cfg.run_id,
                cfg.offline, not cfg.no_judge, cfg.url, cfg.lora, cfg.weight,
                slots=set_designer.parse_slots(cfg.slots),
                max_attempts=max(1, cfg.attempts), resume=cfg.resume)
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
    except Exception as e:
        sys.exit(f"\nRun stopped: {e}")


if __name__ == "__main__":
    main()
