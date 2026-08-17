# Generation stage: prompts.csv → card artwork

Takes what `set_designer.py` produced and renders it through ComfyUI with the LoRA and weight
fixed by the two validation runs, several seeds per card so there is something to choose from.

## Usage

```bash
# how many images and therefore how long
python batch_generate.py --set-dir ../runs/2026-08-14_kino-sets/noir --dry-run

# render the whole set: 3 variants × 10 cards × 3 seeds
python batch_generate.py --set-dir ../runs/2026-08-14_kino-sets/noir --seeds-per-card 3

# one variant only
python batch_generate.py --csv ../runs/2026-08-14_kino-sets/noir/variant_1/prompts.csv

# rebuild contact sheets from images already on disk, no GPU
python batch_generate.py --set-dir ../runs/2026-08-14_kino-sets/noir --sheets-only
```

Defaults are the frozen decisions: `Icon_3D_Flux.safetensors` at weight 0.8, 832×928,
20 steps, guidance 3.5, `fp8_e4m3fn`. Override with `--lora --weight --steps` and so on.

## Output

Written next to each variant, alongside the `cards.json` and `prompts.csv` it came from:

```
variant_N/
├── cards.json      (input, from set_designer)
├── prompts.csv     (input)
├── raw/            every generation: <card_id>__s<seed>.png
├── contact.png     rows are cards, columns are seeds; labels carry name, category, rarity
└── manifest.json   parameters, per-card image list, metrics
```

The contact sheet is the review artefact: ten cards down, seeds across, so a set can be judged
as a set rather than card by card. Its title and subtitle come from `cards.json`, so the
variant's concept sits above the grid it produced.

## Resumability

Existing files are skipped. An interrupted run continues where it stopped, and adding
`--seeds-per-card 4` after a 3-seed run renders only the fourth seed. Seeds are
`--base-seed + k`, so they are stable across runs rather than random.

This matters more than it sounds: a full set at three seeds is 90 images, which on an 8 GB
card is a couple of hours. Losing that to a crash or a reboot would be expensive.

## Metrics

`manifest.json` records what is measurable at this stage: `planned`, `rendered`, `skipped`,
`failed`, `on_disk`, `wall_time_sec`.

`first_try_pass_rate` and `needs_review` from the repository README are deliberately left as
`metrics_pending`. They require the VLM judge — layer 4 — which does not exist yet. Reporting
a pass rate without a judge would be inventing a number.

## Driving a graph authored in the ComfyUI editor

The built-in graph covers plain text-to-image. Anything with a less obvious topology — an
upscale chain, ControlNet, Flux Redux or IP-Adapter style transfer — is far easier to wire
visually than to write blind. Author it in ComfyUI, then hand it to this script:

```bash
python batch_generate.py --set-dir ... --workflow my_style_transfer.json
```

**Export the right format.** Use `Workflow -> Export (API)`, not `Save` or plain `Export`.
The editor format carries node positions and UI state and cannot be submitted to the API; the
script detects it and says so rather than failing obscurely later.

**Mark the fields that vary.** Type these tokens directly into the widget boxes in the editor:

| Token | Goes into |
|---|---|
| `%PROMPT%` | the positive CLIPTextEncode |
| `%NEGATIVE%` | the negative CLIPTextEncode |
| `%SEED%` | the sampler seed |
| `%LORA%` / `%LORA_WEIGHT%` | LoRA loader name and strength |
| `%WIDTH%` `%HEIGHT%` `%STEPS%` `%GUIDANCE%` | latent size and sampler settings |
| `%PREFIX%` | SaveImage `filename_prefix` |

Substitution is by explicit token rather than by inferring node roles. A graph with two
`CLIPTextEncode` nodes offers no reliable way to tell the positive from the negative, and a
wrong guess yields plausible images that quietly ignore half the prompt.

A field containing nothing but a token keeps its type: `%SEED%` becomes the integer `1000`,
not the string `"1000"`, so numeric widgets survive the round trip.

**Check before committing hours.** `--dry-run` prints the tokens found in the graph, the ones
that would actually be substituted, and warnings — a missing `%PROMPT%` (the graph would
render whatever text is baked into the node and ignore the cards), a missing `SaveImage` node,
or tokens the script has no value for. If `%SEED%` is absent, every `seed` and `noise_seed`
field is patched anyway and the run says so, because otherwise a multi-seed batch would render
the same image N times.

Model loaders inside a template are left alone — auto-detection is skipped, since the graph
already states what it wants.

## Shared client

Both this script and `lora_grid_test.py` talk to ComfyUI through `comfy_client.py`:
the API client, the Flux graph, model auto-detection and the contact-sheet builder. It was
extracted once the second consumer appeared, so a fix to the graph lands in one place rather
than two.

`build_flux_graph` pins `cfg` to 1.0 on purpose — Flux takes prompt strength through
`FluxGuidance`, and raising classifier-free guidance on top of it burns the image.
