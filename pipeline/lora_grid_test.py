from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from comfy_client import Comfy, build_flux_graph, build_sheet, pick


TRIGGER = "LORACCG"

# Matches the dataset aspect ratio (512x571 ~ 0.897) and stays a multiple of 32 for Flux.
DEFAULT_W, DEFAULT_H = 832, 928


EPOCH_RE = re.compile(r"[-_](\d{3,})$")


def epoch_num(name: str) -> int | None:
    """Epoch number from the tail of a filename: cardcollectionsg1-000004 -> 4.
    The final checkpoint without a suffix (cardcollectionsg1.safetensors) -> None."""
    m = EPOCH_RE.search(Path(name).stem)
    return int(m.group(1)) if m else None


def epoch_key(name: str) -> tuple:
    """Natural sort order: ep4 < ep12 < final."""
    if name.startswith("ep") and name[2:].isdigit():   # already a label, not a filename
        return (int(name[2:]), name)
    n = epoch_num(name)
    return (10**9 if n is None else n, name)


def epoch_label(name: str) -> str:
    if name == "final" or (name.startswith("ep") and name[2:].isdigit()):
        return name
    n = epoch_num(name)
    return f"ep{n}" if n is not None else "final"


def short_label(name: str) -> str:
    """Row label for an arbitrary third-party LoRA file, e.g.
    'flux/mobile-kid-game-style.safetensors' -> 'mobile-kid-g'."""
    stem = Path(name).stem
    stem = re.sub(r"(?i)(flux[.\-_]?1?[.\-_]?d(ev)?|lora|v\d+)", "", stem)
    stem = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-")
    return (stem[:12] or Path(name).stem[:12])


def parse_lora_specs(raw: str, available: list[str]) -> list[tuple[str, str]]:
    """Parse --loras into [(filename, trigger)].

    Entries are comma-separated; an optional trigger follows '=':
        a.safetensors=SBG_quality,b.safetensors=appcartoongame,c.safetensors
    A missing trigger means the LoRA needs no trigger word.
    """
    out, missing = [], []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, trigger = chunk.partition("=")
        name, trigger = name.strip(), trigger.strip()
        if name not in available:
            # tolerate a bare filename when ComfyUI reports it inside a subfolder
            matches = [a for a in available if Path(a).name == name]
            if len(matches) == 1:
                name = matches[0]
            else:
                missing.append(name)
                continue
        out.append((name, trigger))
    if missing:
        sys.exit(
            "ComfyUI cannot see these LoRA files: " + ", ".join(missing) +
            "\nRun with --list to see the available names."
        )
    return out


def dedupe_labels(pairs: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """[(file, trigger)] -> [(file, label, trigger)] with unique labels."""
    seen: dict[str, int] = {}
    out = []
    for name, trigger in pairs:
        base = epoch_label(name) if epoch_num(name) is not None else short_label(name)
        if base in seen:
            seen[base] += 1
            base = f"{base}~{seen[base]}"
        else:
            seen[base] = 1
        out.append((name, base, trigger))
    return out


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_graph(cfg, lora_name: str | None, strength: float, prompt: str, seed: int) -> dict:
    """Thin adapter: this script carries its settings on an argparse namespace,
    comfy_client.build_flux_graph takes them explicitly."""
    models = {"unet": cfg.unet, "clip1": cfg.clip1, "clip2": cfg.clip2,
              "vae": cfg.vae, "dtype": cfg.dtype}
    return build_flux_graph(
        models, prompt, seed,
        lora_name=lora_name, lora_weight=strength,
        width=cfg.width, height=cfg.height,
        steps=cfg.steps, guidance=cfg.guidance,
        sampler=cfg.sampler, scheduler=cfg.scheduler,
        filename_prefix="loratest/run",
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--out", default="lora_test_out", help="output directory")
    ap.add_argument("--prompts", default="control_prompts.json")
    ap.add_argument("--lora-filter", default="", help="substring to match LoRA file names")
    ap.add_argument("--loras", default="",
                    help="explicit comma-separated LoRA files, each with an optional trigger word: "
                         "'a.safetensors=SBG_quality,b.safetensors=appcartoongame,c.safetensors'. "
                         "Use this to compare different third-party LoRAs, which rarely share a trigger.")
    ap.add_argument("--weights", default="0,0.6,0.8,1.0", help="comma-separated LoRA weights; 0 = base model")
    ap.add_argument("--seeds", default="101", help="comma-separated seeds")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--width", type=int, default=DEFAULT_W)
    ap.add_argument("--height", type=int, default=DEFAULT_H)
    ap.add_argument("--trigger", default=TRIGGER,
                    help="trigger word applied when LoRAs are selected with --lora-filter; "
                         "ignored for --loras entries, which carry their own")
    ap.add_argument("--unet", default="", help="UNET file name (auto-detected by default)")
    ap.add_argument("--clip1", default="", help="t5xxl")
    ap.add_argument("--clip2", default="", help="clip_l")
    ap.add_argument("--vae", default="")
    ap.add_argument("--dtype", default="", help="weight_dtype for UNETLoader, e.g. fp8_e4m3fn")
    ap.add_argument("--list", action="store_true", help="print the models ComfyUI can see and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the run plan and exit")
    ap.add_argument("--sheets-only", action="store_true", help="only rebuild the contact sheets")
    ap.add_argument("--timeout", type=int, default=600, help="per-image timeout, seconds")
    cfg = ap.parse_args()

    out = Path(cfg.out)
    img_dir = out / "images"
    prompts_path = Path(cfg.prompts)
    if not prompts_path.exists():
        sys.exit(f"Prompt file not found: {prompts_path}")
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))

    weights = [float(w) for w in cfg.weights.split(",") if w.strip() != ""]
    seeds = [int(s) for s in cfg.seeds.split(",") if s.strip() != ""]

    # ---------------- sheets-only: assemble from whatever is already on disk ---
    if cfg.sheets_only:
        tags = {p.stem.split("__")[0] for p in img_dir.glob("*.png")} - {"base"}
        loras = sorted(tags, key=epoch_key)
        make_sheets(out, img_dir, prompts, loras, weights, seeds)
        return

    api = Comfy(cfg.url, timeout=cfg.timeout)
    api.ping()

    unet_opts = api.options("UNETLoader", "unet_name")
    clip_opts = api.options("DualCLIPLoader", "clip_name1")
    vae_opts = api.options("VAELoader", "vae_name")
    lora_opts = api.options("LoraLoaderModelOnly", "lora_name")
    dtype_opts = api.options("UNETLoader", "weight_dtype")

    if cfg.list:
        for label, opts in (
            ("UNET", unet_opts), ("CLIP", clip_opts), ("VAE", vae_opts),
            ("LoRA", lora_opts), ("weight_dtype", dtype_opts),
        ):
            print(f"\n=== {label} ({len(opts)}) ===")
            for o in opts:
                print("  ", o)
        return

    cfg.unet = cfg.unet or pick(unet_opts, "flux1-dev", "flux", label="UNET")
    cfg.clip1 = cfg.clip1 or pick(clip_opts, "t5", label="CLIP t5")
    cfg.clip2 = cfg.clip2 or pick(clip_opts, "clip_l", "clip-l", label="CLIP L")
    cfg.vae = cfg.vae or pick(vae_opts, "ae.", "flux", label="VAE")
    # fp8 is the sane default on consumer cards: bf16 Flux is 24 GB and thrashes on 8-12 GB
    # GPUs. Override with --dtype default for maximum quality on a large card.
    cfg.dtype = cfg.dtype or (pick(dtype_opts, "fp8_e4m3fn", label="weight_dtype")
                              if dtype_opts else "default")

    if cfg.loras:
        pairs = parse_lora_specs(cfg.loras, lora_opts)
    else:
        pairs = [(x, cfg.trigger) for x in lora_opts if cfg.lora_filter.lower() in x.lower()]
        pairs.sort(key=lambda p: epoch_key(p[0]))
    if not pairs:
        sys.exit(
            "No LoRA file matched the filter.\n"
            "Run with --list to see what is actually in models/loras."
        )
    entries = dedupe_labels(pairs)          # [(file, label, trigger)]
    loras = [e[1] for e in entries]         # labels, used for rows and file names

    # ---------------- run plan ------------------------------------------------
    # Order matters for wall-clock, not just tidiness: ComfyUI keeps the patched model in
    # memory between jobs, but changing lora_name forces a reload from disk. Looping LoRA
    # first and prompts innermost means one load per LoRA instead of one per image.
    jobs = []          # (lora_file_or_None, weight, prompt_text, seed, out_path)
    seen: set = set()
    for lora_file, label, trigger in entries:
        for w in weights:
            for pr in prompts:
                for sd in seeds:
                    # weight 0 means no LoRA at all, so it is identical across rows
                    # and its trigger word would be meaningless — render it once, bare.
                    if w == 0:
                        key_lora, tag, trig = None, "base", ""
                    else:
                        key_lora, tag, trig = lora_file, label, trigger
                    text = pr["text"]
                    if trig and pr.get("trigger", True):
                        text = f"{trig} {text}"
                    name = f"{tag}__w{w:g}__{pr['id']}__s{sd}.png"
                    if name in seen:
                        continue
                    seen.add(name)
                    jobs.append((key_lora, w, text, sd, img_dir / name))

    todo = [j for j in jobs if not j[4].exists()]
    lora_summary = ", ".join(
        "{}[{}]".format(lab, trg if trg else "no trigger") for _, lab, trg in entries
    )
    print(
        f"Model:   {cfg.unet}  [{cfg.dtype}]\n"
        f"CLIP:    {cfg.clip1} + {cfg.clip2}\n"
        f"VAE:     {cfg.vae}\n"
        f"LoRA:    {len(entries)} file(s) -> {lora_summary}\n"
        f"Weights: {weights}   Seeds: {seeds}   Prompts: {len(prompts)}\n"
        f"Size:    {cfg.width}x{cfg.height}, steps={cfg.steps}, guidance={cfg.guidance}\n"
        f"Total images: {len(jobs)}, left to render: {len(todo)}"
    )
    if cfg.dry_run:
        for j in todo[:15]:
            print("   ", j[4].name)
        if len(todo) > 15:
            print(f"    ... and {len(todo) - 15} more")
        return

    img_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, (lo, w, text, sd, dest) in enumerate(todo, 1):
        graph = build_graph(cfg, lo, w, text, sd)
        try:
            pid = api.submit(graph)
            entry = api.wait(pid)
        except Exception as e:
            print(f"[{i}/{len(todo)}] {dest.name} — SKIPPED: {e}")
            continue
        images = [im for node in entry.get("outputs", {}).values() for im in node.get("images", [])]
        if not images:
            print(f"[{i}/{len(todo)}] {dest.name} — ComfyUI returned nothing")
            continue
        dest.write_bytes(api.fetch_image(images[0]))
        done = i / max(1, len(todo))
        eta = (time.time() - t0) / max(1e-9, done) * (1 - done)
        print(f"[{i}/{len(todo)}] {dest.name}   ~{eta/60:.1f} min left")

    make_sheets(out, img_dir, prompts, loras, weights, seeds)


def make_sheets(out: Path, img_dir: Path, prompts, loras, weights, seeds) -> None:
    try:
        import PIL  # noqa: F401
    except ImportError:
        print("\nPillow is not installed — images are in", img_dir, "but no sheets were built.")
        print("Install it with: pip install pillow")
        return

    sheet_dir = out / "sheets"
    # `loras` here is already a list of row labels, not file names
    rows = (["base"] if any(w == 0 for w in weights) else []) + list(loras)
    cols = [f"w={w:g}" for w in weights]
    made = 0
    for pr in prompts:
        for sd in seeds:
            cells = {}
            for r in rows:
                for w in weights:
                    tag = "base" if w == 0 else r
                    if r == "base" and w != 0:
                        continue
                    if r != "base" and w == 0:
                        continue
                    cells[(r, f"w={w:g}")] = img_dir / f"{tag}__w{w:g}__{pr['id']}__s{sd}.png"
            title = f"{pr['id']}   seed={sd}"
            if build_sheet(sheet_dir / f"sheet_{pr['id']}_s{sd}.png", cells, rows, cols,
                           title, pr.get("note", "")):
                made += 1
    print(f"\nDone. Images: {img_dir}\nContact sheets ({made}): {sheet_dir}")


if __name__ == "__main__":
    main()
