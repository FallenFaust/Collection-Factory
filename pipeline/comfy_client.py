#!/usr/bin/env python3
"""
comfy_client.py — shared ComfyUI API client and Flux graph builder.

Extracted from lora_grid_test.py once a second script needed the same plumbing.
Both the validation grid and the batch generator talk to ComfyUI through this module,
so a fix to the graph or the polling logic lands in one place.

Not a general-purpose ComfyUI library: it covers exactly the Flux text-to-image path
this pipeline uses.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8188"
NEGATIVE_DEFAULT = ""


class Comfy:
    """Minimal client over the ComfyUI HTTP API."""

    def __init__(self, base: str = DEFAULT_URL, timeout: int = 600):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())
        self._object_info: dict | None = None

    # ---------------- transport ------------------------------------------- #
    def _get(self, path: str) -> bytes:
        with urllib.request.urlopen(f"{self.base}{path}", timeout=30) as r:
            return r.read()

    def _get_json(self, path: str):
        return json.loads(self._get(path).decode("utf-8"))

    def ping(self) -> None:
        try:
            self._get_json("/system_stats")
        except Exception as e:
            sys.exit(
                f"Cannot reach ComfyUI at {self.base} ({e}).\n"
                "Start ComfyUI, check the port, or pass --url.\n"
                "The Desktop build sometimes picks a free port instead of its default."
            )

    # ---------------- introspection --------------------------------------- #
    def object_info(self) -> dict:
        if self._object_info is None:
            self._object_info = self._get_json("/object_info")
        return self._object_info

    def options(self, node: str, field: str) -> list[str]:
        """Allowed values of a combo field on a node — model file names and the like."""
        info = self.object_info().get(node)
        if not info:
            return []
        for section in ("required", "optional"):
            spec = info.get("input", {}).get(section, {}).get(field)
            if spec and isinstance(spec[0], list):
                return [str(x) for x in spec[0]]
        return []

    # ---------------- execution ------------------------------------------- #
    def submit(self, graph: dict) -> str:
        payload = json.dumps({"prompt": graph, "client_id": self.client_id}).encode()
        req = urllib.request.Request(
            f"{self.base}/prompt", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))["prompt_id"]
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"ComfyUI rejected the graph ({e.code}):\n{body}") from None

    def wait(self, prompt_id: str) -> dict:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            hist = self._get_json(f"/history/{prompt_id}")
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                    detail = msgs[0][1].get("exception_message") if msgs else "unknown error"
                    raise RuntimeError(f"Execution failed: {detail}")
                return entry
            time.sleep(1.0)
        raise TimeoutError(f"Job {prompt_id} did not finish within {self.timeout}s")

    def fetch_image(self, meta: dict) -> bytes:
        q = urllib.parse.urlencode({
            "filename": meta["filename"],
            "subfolder": meta.get("subfolder", ""),
            "type": meta.get("type", "output"),
        })
        return self._get(f"/view?{q}")

    def render(self, graph: dict) -> bytes | None:
        """submit + wait + fetch the first output image. None if nothing came back."""
        entry = self.wait(self.submit(graph))
        images = [im for node in entry.get("outputs", {}).values() for im in node.get("images", [])]
        return self.fetch_image(images[0]) if images else None


# --------------------------------------------------------------------------- #
# Model discovery
# --------------------------------------------------------------------------- #
def pick(options: list[str], *preferences: str, label: str = "") -> str:
    """First option containing one of the preferred substrings, else the first one."""
    if not options:
        sys.exit(f"ComfyUI reports no files for {label}. Check your models/ directories.")
    low = [o.lower() for o in options]
    for pref in preferences:
        for i, name in enumerate(low):
            if pref in name:
                return options[i]
    return options[0]


def resolve_models(api: Comfy, unet: str = "", clip1: str = "", clip2: str = "",
                   vae: str = "", dtype: str = "") -> dict:
    """Fill in whatever was not passed explicitly by asking ComfyUI what exists.

    fp8 is the default weight_dtype on purpose: bf16 Flux is a 24 GB model and thrashes
    on an 8-12 GB card. Pass dtype='default' on a large GPU for maximum quality.
    """
    dtypes = api.options("UNETLoader", "weight_dtype")
    return {
        "unet": unet or pick(api.options("UNETLoader", "unet_name"),
                             "flux1-dev", "flux", label="UNET"),
        "clip1": clip1 or pick(api.options("DualCLIPLoader", "clip_name1"), "t5", label="CLIP t5"),
        "clip2": clip2 or pick(api.options("DualCLIPLoader", "clip_name1"),
                               "clip_l", "clip-l", label="CLIP L"),
        "vae": vae or pick(api.options("VAELoader", "vae_name"), "ae.", "flux", label="VAE"),
        "dtype": dtype or (pick(dtypes, "fp8_e4m3fn", label="weight_dtype") if dtypes else "default"),
    }


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_flux_graph(models: dict, prompt: str, seed: int, *,
                     negative: str = NEGATIVE_DEFAULT,
                     lora_name: str | None = None, lora_weight: float = 0.0,
                     width: int = 832, height: int = 928,
                     steps: int = 20, guidance: float = 3.5,
                     sampler: str = "euler", scheduler: str = "simple",
                     filename_prefix: str = "cardgen/run") -> dict:
    """Flux text-to-image graph, optionally with one LoRA applied to the model only.

    cfg is pinned to 1.0 because Flux uses FluxGuidance for prompt strength instead of
    classifier-free guidance; raising cfg here would just burn the image.
    """
    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": models["unet"], "weight_dtype": models["dtype"]}},
        "2": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": models["clip1"], "clip_name2": models["clip2"],
                         "type": "flux"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": models["vae"]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": {"class_type": "FluxGuidance",
              "inputs": {"conditioning": ["5", 0], "guidance": guidance}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative}},
        "7": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["6", 0], "negative": ["9", 0],
                         "latent_image": ["7", 0], "seed": seed, "steps": steps, "cfg": 1.0,
                         "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0], "filename_prefix": filename_prefix}},
    }
    if lora_name and lora_weight > 0:
        g["4"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["1", 0], "lora_name": lora_name,
                             "strength_model": lora_weight}}
        g["8"]["inputs"]["model"] = ["4", 0]
    return g


# --------------------------------------------------------------------------- #
# Workflow templates exported from the ComfyUI editor
# --------------------------------------------------------------------------- #
#
# build_flux_graph above covers the plain text-to-image path. Anything with a less obvious
# topology — an upscale chain, ControlNet, Flux Redux or IP-Adapter style transfer — is far
# easier to wire visually than to write blind. For those, author the graph in ComfyUI, save it
# with Workflow -> Export (API), and drive it from here.
#
# Substitution is by explicit placeholder rather than by guessing node roles: a graph with two
# CLIPTextEncode nodes gives no reliable way to tell the positive from the negative, and a
# wrong guess produces plausible images that quietly ignore half the prompt.
#
# Type the tokens straight into the widget fields in the editor:
#
#   %PROMPT%       positive prompt text
#   %NEGATIVE%     negative prompt text
#   %SEED%         sampler seed
#   %LORA%         LoRA file name
#   %LORA_WEIGHT%  LoRA strength
#   %WIDTH% %HEIGHT% %STEPS% %GUIDANCE%
#   %PREFIX%       SaveImage filename_prefix
#
# A field holding nothing but a token keeps its type — %SEED% becomes the integer 1000, not
# the string "1000" — so numeric widgets survive the round trip.

TOKEN_RE = re.compile(r"%([A-Z_]+)%")


def load_workflow_template(path: str | Path) -> dict:
    """Read an API-format workflow. Rejects the editor format, which looks similar but is not."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        sys.exit(f"Cannot read workflow {p}: {e}")
    if not isinstance(data, dict):
        sys.exit(f"{p} is not a JSON object.")
    if "nodes" in data and "links" in data:
        sys.exit(
            f"{p} is the editor format, not the API format.\n"
            "In ComfyUI use Workflow -> Export (API), not Save / Export."
        )
    if not any(isinstance(v, dict) and "class_type" in v for v in data.values()):
        sys.exit(f"{p} has no nodes with class_type — is it really an exported workflow?")
    return data


def template_placeholders(tmpl: dict) -> set[str]:
    found: set[str] = set()

    def walk(node):
        if isinstance(node, str):
            found.update(TOKEN_RE.findall(node))
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(tmpl)
    return found


def apply_template(tmpl: dict, values: dict) -> tuple[dict, set[str]]:
    """Return a copy of the workflow with placeholders replaced, plus the tokens used."""
    used: set[str] = set()

    def convert(s: str):
        whole = TOKEN_RE.fullmatch(s.strip())
        if whole and whole.group(1) in values:
            used.add(whole.group(1))
            return values[whole.group(1)]          # keep the value's own type

        def sub(m):
            key = m.group(1)
            if key in values:
                used.add(key)
                return str(values[key])
            return m.group(0)                      # unknown token: leave it alone

        return TOKEN_RE.sub(sub, s)

    def walk(node):
        if isinstance(node, str):
            return convert(node)
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    graph = walk(tmpl)

    # Convenience: a graph without %SEED% would silently render the same image every time,
    # which is the opposite of what a multi-seed batch wants. Patch every sampler seed field.
    if "SEED" not in used and "SEED" in values:
        for node in graph.values():
            if not isinstance(node, dict):
                continue
            for field in ("seed", "noise_seed"):
                if field in node.get("inputs", {}):
                    node["inputs"][field] = values["SEED"]
                    used.add("SEED")
    return graph, used


def check_template(tmpl: dict, offered: dict) -> list[str]:
    """Warnings worth printing before a run that will take hours."""
    found = template_placeholders(tmpl)
    warn = []
    if "PROMPT" not in found:
        warn.append("no %PROMPT% — the graph will render whatever is hard-wired in the node, "
                    "and the cards' prompts will never reach the generator")
    if "SEED" not in found:
        warn.append("no %SEED% — seeds will be filled into seed/noise_seed fields automatically")
    if not any(isinstance(n, dict) and n.get("class_type") == "SaveImage" for n in tmpl.values()):
        warn.append("no SaveImage node — there will be nothing for the script to collect")
    unused = sorted(found - set(offered))
    if unused:
        warn.append("placeholders in the graph with no value to fill them: " + ", ".join(unused))
    return warn


# --------------------------------------------------------------------------- #
# Contact sheets
# --------------------------------------------------------------------------- #
def load_font(size: int):
    from PIL import ImageFont

    for candidate in (
        "arial.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build_sheet(out_png: Path, cells: dict, rows: list[str], cols: list[str],
                title: str, subtitle: str = "", thumb: int = 384,
                row_sub: dict | None = None) -> bool:
    """Grid of images with labelled axes. cells[(row, col)] = path.

    row_sub optionally carries a second, smaller line under each row label — used by the
    batch generator to show category and rarity beneath the card name.
    """
    import textwrap

    from PIL import Image, ImageDraw

    pad, head, top = 8, 40, 44
    sample = next((p for p in cells.values() if p.exists()), None)
    if sample is None:
        return False
    w0, h0 = Image.open(sample).size
    tw = thumb
    th = int(round(h0 * thumb / w0))

    # The left gutter has to fit the longest row label, otherwise Russian card names
    # run underneath the first thumbnail. Measure instead of guessing.
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    f_probe, f_probe_sub = load_font(19), load_font(15)
    widths = [probe.textlength(r[:24], font=f_probe) for r in rows] or [0]
    if row_sub:
        widths += [probe.textlength(v[:26], font=f_probe_sub) for v in row_sub.values()]
    side = int(max(150, min(360, max(widths) + 2 * pad + 6)))

    W = side + len(cols) * (tw + pad) + pad
    sub_lines = textwrap.wrap(subtitle, width=max(40, W // 11)) if subtitle else []
    top += 24 * len(sub_lines)
    H = top + head + len(rows) * (th + pad) + pad
    sheet = Image.new("RGB", (W, H), (24, 24, 27))
    d = ImageDraw.Draw(sheet)
    f_title, f_label, f_note = load_font(26), load_font(19), load_font(15)

    d.text((pad, pad + 2), title, font=f_title, fill=(240, 240, 245))
    for i, line in enumerate(sub_lines):
        d.text((pad, pad + 34 + i * 24), line, font=f_note, fill=(150, 150, 160))
    for ci, col in enumerate(cols):
        d.text((side + ci * (tw + pad) + 6, top + head - 30), col, font=f_label,
               fill=(190, 190, 200))

    for ri, row in enumerate(rows):
        y = top + head + ri * (th + pad)
        d.text((pad, y + th // 2 - 18), row[:24], font=f_label, fill=(225, 225, 232))
        if row_sub and row in row_sub:
            d.text((pad, y + th // 2 + 6), row_sub[row][:26], font=f_note, fill=(140, 140, 150))
        for ci, col in enumerate(cols):
            p = cells.get((row, col))
            if not p or not p.exists():
                continue
            im = Image.open(p).convert("RGB").resize((tw, th), Image.LANCZOS)
            sheet.paste(im, (side + ci * (tw + pad), y))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png, quality=95)
    return True
