# LoRA selection grid

Renders a matrix of [LoRA] × [weight] × [control prompt] on fixed seeds through the ComfyUI
API and assembles one contact sheet per prompt — rows are LoRAs, columns are weights, and the
top `base` row is the bare base model on the same seed for reference.

It answers two different questions depending on what you feed it:

* **Which epoch of my own LoRA should I keep, and at what weight?**
  Rows are epoch checkpoints of one training run.
* **Which of these third-party LoRAs actually fits my style?**
  Rows are different LoRAs, each with its own trigger word.

The second mode is the one in use here: the project uses public LoRAs rather than a
self-trained one, because Flux LoRA training needs more VRAM and system RAM than the machine
has (see `../claude/02-обучение-lora.md` in the project notes).

## Contents

| File | Purpose |
|---|---|
| `lora_grid_test.py` | Runs the matrix and assembles the contact sheets |
| `control_prompts.json` | 8 control prompts: one per object category from the brief, plus genre stress tests and two overfit detectors |
| `mock_comfy.py` | Fake ComfyUI on port 8199 — exercise the runner without a GPU |

## Usage

```bash
# what ComfyUI can see
python lora_grid_test.py --list

# compare several third-party LoRAs, each with its own trigger word
python lora_grid_test.py --loras "mobile-kid-game-style.safetensors=appcartoongame,casual_game_art.safetensors=SBG_quality,3d_game_icon.safetensors" --weights 0,0.7,0.9 --dry-run

# epochs of one training run, one shared trigger
python lora_grid_test.py --lora-filter mylora --trigger MYTRIGGER --weights 0,0.6,0.8,1.0
```

In `--loras`, everything after `=` is that LoRA's trigger word; omit it entirely for a LoRA
that needs none. Third-party LoRAs almost never share a trigger, which is why `--trigger`
applies only to `--lora-filter` mode. The `base` row is always rendered without any trigger,
since there is no LoRA loaded to respond to it.

Output: `lora_test_out/images/` holds the raw PNGs, `lora_test_out/sheets/` holds the sheets.

Worth knowing:

* The run is **resumable** — existing files are skipped, so it can be interrupted and continued.
* `--sheets-only` rebuilds the sheets from images already on disk, without touching the GPU.
* `--seeds 101,202,303` renders three seeds instead of one, which separates "this LoRA is bad"
  from "that seed was unlucky".
* UNET, CLIP and VAE are auto-detected; override with `--unet --clip1 --clip2 --vae`.
* `weight_dtype` defaults to **fp8_e4m3fn**, because bf16 Flux is a 24 GB model and thrashes on
  an 8-12 GB card. Pass `--dtype default` on a large GPU for maximum quality.
* Default size is 832×928 — a portrait card ratio, rounded to a multiple of 32.

Cost: 8 prompts × 3 LoRAs × 2 non-zero weights + 8 base images = 56 renders. On an 8 GB card at
20 steps that is roughly an hour. Trim `control_prompts.json` to the four category prompts to
halve it.

## Reading the sheets

**Fitting the style.** Read down the strongest weight column. What matters is not which image
is prettiest in isolation but which row could plausibly sit in the same card set as the
others — consistent light, consistent level of detail, consistent silhouette weight.

**The categories are the real test.** Four of the prompts correspond to the four object
categories the brief requires in every set. A LoRA that renders beautiful cluttered scenes but
cannot leave a background empty will fail categories 1 and 2, and no prompt engineering will
rescue it. Check `cat1_float_gradient` first — it is the one most style LoRAs fail, because
they are trained on full illustrations.

**Genre range.** `cat4_genre_scifi` and `cat1_genre_horror` are night, neon and dark palettes.
The target collection needs horror, western and sci-fi. A LoRA locked into warm daylight will
show it here, and that is expensive to discover later.

**Overfitting and leakage.** `overfit_minimal` asks for an apple on a plain background: a LoRA
that adds scenery invented the background rather than the style. `notrigger_control` is the
same prompt without the trigger word — if it looks identical, the trigger does nothing and the
LoRA overrides the base model wholesale, which means you cannot dial the style down.

**Choosing the weight.** Low weights leave base-model realism showing through; at 1.0 most
style LoRAs are overcooked — contrast maxed, small details turning to mush. The working point
is usually 0.7-0.9. If the style only appears at 1.0, the LoRA is too weak for the job.

## Next

The chosen LoRA and weight feed the generation stage. `README_set_designer.md` covers the
stage that produces the prompts.
