#!/usr/bin/env python3
"""
batch_generate.py — generation stage: prompts.csv -> card artwork.

Takes the output of set_designer.py and renders every card through ComfyUI with the LoRA
and weight fixed by the validation runs, several seeds per card so a human (later, a VLM
judge) has something to choose from.

Writes next to each variant:
    raw/<card_id>__s<seed>.png   every generation
    contact.png                  one sheet: rows are cards, columns are seeds
    manifest.json                parameters, per-card seeds, metrics

Resumable: existing files are skipped, so an interrupted run continues where it stopped.

Examples:
    python batch_generate.py --set-dir ../runs/2026-08-14_kino-sets/noir --dry-run
    python batch_generate.py --set-dir ../runs/2026-08-14_kino-sets/noir --seeds-per-card 3
    python batch_generate.py --csv path/to/prompts.csv --seeds-per-card 2
    python batch_generate.py --set-dir ... --sheets-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from comfy_client import (Comfy, apply_template, build_flux_graph, build_sheet,
                          check_template, load_workflow_template, resolve_models,
                          template_placeholders)

DEFAULT_LORA = "Icon_3D_Flux.safetensors"
DEFAULT_WEIGHT = 0.8


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def read_cards(csv_path: Path) -> list[dict]:
    # set_designer writes utf-8-sig so Excel opens the file without mangling Cyrillic
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"card_id", "name", "category", "rarity", "prompt"}
    if not rows:
        sys.exit(f"{csv_path} is empty")
    missing = required - set(rows[0])
    if missing:
        sys.exit(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")
    return rows


def template_values(card: dict, seed: int, cfg) -> dict:
    """Values offered to a workflow template for one card and seed."""
    return {
        "PROMPT": card["prompt"],
        "NEGATIVE": card.get("negative", ""),
        "SEED": seed,
        "LORA": cfg.lora,
        "LORA_WEIGHT": cfg.weight,
        "WIDTH": cfg.width,
        "HEIGHT": cfg.height,
        "STEPS": cfg.steps,
        "GUIDANCE": cfg.guidance,
        "PREFIX": f"cardgen/{card['card_id']}",
    }


def find_variants(set_dir: Path) -> list[Path]:
    found = sorted(set_dir.glob("variant_*/prompts.csv"))
    if not found:
        sys.exit(
            f"No variant_*/prompts.csv under {set_dir}.\n"
            "Point --set-dir at a set directory produced by set_designer.py, "
            "or pass a single file with --csv."
        )
    return found


# --------------------------------------------------------------------------- #
# One variant
# --------------------------------------------------------------------------- #
def output_dir(csv_path: Path, cfg) -> Path:
    """Where results go. By default next to the prompts; --out redirects them elsewhere,
    which is what makes an A/B of two LoRAs on the same set possible without collisions."""
    if not cfg.out:
        return csv_path.parent
    if cfg.csv:
        return Path(cfg.out) / csv_path.parent.name
    return Path(cfg.out) / csv_path.parent.parent.name / csv_path.parent.name


def render_variant(api: Comfy | None, models: dict, csv_path: Path, cfg) -> dict:
    vdir = output_dir(csv_path, cfg)
    vdir.mkdir(parents=True, exist_ok=True)
    raw = vdir / "raw"
    cards = read_cards(csv_path)
    seeds = [cfg.base_seed + k for k in range(cfg.seeds_per_card)]

    jobs = []
    for card in cards:
        for sd in seeds:
            jobs.append((card, sd, raw / f"{card['card_id']}__s{sd}.png"))
    todo = [j for j in jobs if not j[2].exists()]

    label = f"{vdir.parent.name}/{vdir.name}"
    print(f"\n=== {label}: {len(cards)} cards × {len(seeds)} seeds = {len(jobs)}, "
          f"{len(todo)} left")

    stats = {"planned": len(jobs), "rendered": 0, "skipped": len(jobs) - len(todo), "failed": 0}

    if cfg.dry_run or cfg.sheets_only:
        return finish_variant(vdir, csv_path.parent, cards, seeds, stats, cfg,
                              wall=0.0, write=not cfg.dry_run)

    raw.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, (card, sd, dest) in enumerate(todo, 1):
        if cfg.template is not None:
            graph, _ = apply_template(cfg.template,
                                      template_values(card, sd, cfg))
        else:
            graph = build_flux_graph(
                models, card["prompt"], sd,
                negative=card.get("negative", ""),
                lora_name=cfg.lora, lora_weight=cfg.weight,
                width=cfg.width, height=cfg.height,
                steps=cfg.steps, guidance=cfg.guidance,
                filename_prefix=f"cardgen/{card['card_id']}",
            )
        try:
            data = api.render(graph)
        except Exception as e:
            print(f"[{i}/{len(todo)}] {dest.name} — SKIPPED: {e}")
            stats["failed"] += 1
            continue
        if not data:
            print(f"[{i}/{len(todo)}] {dest.name} — ComfyUI returned nothing")
            stats["failed"] += 1
            continue
        dest.write_bytes(data)
        stats["rendered"] += 1
        # Hooks for a caller that is not a terminal. The orchestrator uses them to drive a
        # progress bar and to honour a cancel button; nothing here changes for the CLI, which
        # simply does not set them.
        if getattr(cfg, "on_image", None):
            cfg.on_image(i, len(todo), dest)
        if getattr(cfg, "should_stop", None) and cfg.should_stop():
            stats["stopped"] = True
            break
        done = i / max(1, len(todo))
        eta = (time.time() - t0) / max(1e-9, done) * (1 - done)
        print(f"[{i}/{len(todo)}] {card['name']} s{sd}   ~{eta/60:.0f} min left")

    return finish_variant(vdir, csv_path.parent, cards, seeds, stats, cfg,
                          wall=time.time() - t0)


def finish_variant(vdir: Path, src_dir: Path, cards, seeds, stats, cfg,
                   wall: float, write: bool = True) -> dict:
    raw = vdir / "raw"
    rows = [f"{c['n']}. {c['name']}" for c in cards]
    row_sub = {f"{c['n']}. {c['name']}": f"cat {c['category']} · {c['rarity']}" for c in cards}
    cols = [f"seed {s}" for s in seeds]
    cells = {}
    for c in cards:
        for s in seeds:
            cells[(f"{c['n']}. {c['name']}", f"seed {s}")] = raw / f"{c['card_id']}__s{s}.png"

    present = sum(1 for p in cells.values() if p.exists())
    stats["on_disk"] = present
    stats["wall_time_sec"] = round(wall, 1)

    if write and present:
        meta_path = src_dir / "cards.json"
        title = src_dir.parent.name + " / " + src_dir.name
        subtitle = ""
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            title = f"{meta.get('set_title', title)}  ({vdir.name})"
            subtitle = meta.get("concept", "")
        build_sheet(vdir / "contact.png", cells, rows, cols, title, subtitle,
                    thumb=cfg.thumb, row_sub=row_sub)

        manifest = {
            "variant_dir": str(vdir),
            "generation": {
                "lora": cfg.lora, "weight": cfg.weight,
                "width": cfg.width, "height": cfg.height,
                "steps": cfg.steps, "guidance": cfg.guidance,
                "seeds": seeds,
            },
            "cards": [
                {"card_id": c["card_id"], "n": int(c["n"]), "name": c["name"],
                 "category": int(c["category"]), "rarity": c["rarity"],
                 "images": [f"raw/{c['card_id']}__s{s}.png" for s in seeds
                            if (raw / f"{c['card_id']}__s{s}.png").exists()]}
                for c in cards
            ],
            "metrics": stats,
            "metrics_pending": {
                "first_try_pass_rate": "written by the judge stage; run judge.py or the orchestrator",
                "needs_review": "the same",
            },
        }
        (vdir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    -> contact.png, manifest.json  ({present}/{stats['planned']} images)")
    return stats


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--set-dir", help="a set directory containing variant_* subfolders")
    src.add_argument("--csv", help="a single prompts.csv file")

    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--workflow", default="",
                    help="a graph exported from ComfyUI via Workflow -> Export (API). "
                         "The placeholders %%PROMPT%% %%SEED%% %%LORA%% %%LORA_WEIGHT%% "
                         "%%NEGATIVE%% %%WIDTH%% %%HEIGHT%% %%STEPS%% %%GUIDANCE%% %%PREFIX%% "
                         "are substituted per card. Without the flag the built-in "
                         "text-to-image graph is used")
    ap.add_argument("--lora", default=DEFAULT_LORA)
    ap.add_argument("--weight", type=float, default=DEFAULT_WEIGHT)
    ap.add_argument("--out", default="",
                    help="where results go; next to the prompts by default. Use it to run one "
                    "set through two LoRAs without the second overwriting the first")
    ap.add_argument("--seeds-per-card", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=928)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--thumb", type=int, default=256, help="thumbnail width in the contact sheet")
    ap.add_argument("--unet", default="")
    ap.add_argument("--clip1", default="")
    ap.add_argument("--clip2", default="")
    ap.add_argument("--vae", default="")
    ap.add_argument("--dtype", default="")
    ap.add_argument("--timeout", type=int, default=600, help="per-image timeout, seconds")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--sheets-only", action="store_true",
                    help="rebuild contact sheets from existing images, without touching the GPU")
    cfg = ap.parse_args()

    if cfg.csv:
        csvs = [Path(cfg.csv)]
        if not csvs[0].exists():
            sys.exit(f"{csvs[0]} not found")
    else:
        csvs = find_variants(Path(cfg.set_dir))

    # ---------------- workflow template ---------------------------------------
    cfg.template, cfg.tpl_tokens = None, set()
    if cfg.workflow:
        cfg.template = load_workflow_template(cfg.workflow)
        probe = read_cards(csvs[0])[0]
        offered = template_values(probe, cfg.base_seed, cfg)
        found = cfg.tpl_tokens = template_placeholders(cfg.template)
        print(f"Graph: {cfg.workflow}, {len(cfg.template)} nodes")
        print("  placeholders: " + (", ".join(f"%{t}%" for t in sorted(found)) or "none found"))
        for w in check_template(cfg.template, offered):
            print("  WARNING: " + w)
        # a dry run should prove the substitution actually happens, not just that tokens exist
        if cfg.dry_run:
            _, used = apply_template(cfg.template, offered)
            print("  would substitute: " + ", ".join(f"%{t}%" for t in sorted(used)))

    api, models = None, {}
    if not (cfg.dry_run or cfg.sheets_only):
        api = Comfy(cfg.url, timeout=cfg.timeout)
        api.ping()

        # The LoRA name only has to exist when something will actually consume it: either the
        # built-in graph, or a template that contains %LORA%. A template with its own loader
        # node hard-wired needs no check here.
        if cfg.template is None or "LORA" in cfg.tpl_tokens:
            loras = api.options("LoraLoaderModelOnly", "lora_name")
            if cfg.lora not in loras:
                match = [x for x in loras if Path(x).name == cfg.lora]
                if len(match) == 1:
                    cfg.lora = match[0]
                else:
                    sys.exit(f"ComfyUI cannot see the LoRA '{cfg.lora}'.\nAvailable: " +
                             ", ".join(loras[:20]))

        if cfg.template is None:
            models = resolve_models(api, cfg.unet, cfg.clip1, cfg.clip2, cfg.vae, cfg.dtype)
            print(f"Model: {models['unet']} [{models['dtype']}]   "
                  f"LoRA: {cfg.lora} @ {cfg.weight}")
        else:
            # The template carries its own loaders; auto-detection would only mislead.
            print(f"Models come from the graph. LoRA: {cfg.lora} @ {cfg.weight}"
                  if "LORA" in cfg.tpl_tokens else "Models and LoRA come from the graph.")
        print(f"Size: {cfg.width}x{cfg.height}, steps={cfg.steps}, guidance={cfg.guidance}, "
              f"seeds per card: {cfg.seeds_per_card}")

    totals = {"planned": 0, "rendered": 0, "skipped": 0, "failed": 0}
    t0 = time.time()
    for path in csvs:
        s = render_variant(api, models, path, cfg)
        for k in totals:
            totals[k] += s.get(k, 0)

    print(f"\nTotal: planned {totals['planned']}, rendered {totals['rendered']}, "
          f"skipped as already done {totals['skipped']}, failed {totals['failed']}, "
          f"time {(time.time()-t0)/60:.0f} min")
    if cfg.dry_run:
        print("That was a --dry-run; nothing was rendered.")


if __name__ == "__main__":
    main()
