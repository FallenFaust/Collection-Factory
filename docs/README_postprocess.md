# Post-processing: rebuilding category-1 backgrounds

Category 1 in the brief is an object hovering in the air over a flat colour, a gradient or a
simple pattern. Flux will not produce that. It puts a cast shadow under the object and, until
the prompt was rewritten, a floor for the shadow to fall on.

## What was tried first

| Attempt | Result |
|---|---|
| `--negative "drop shadow, cast shadow, ..."` | No effect. The graph pins `cfg` to 1.0, which is correct for a guidance-distilled model but means classifier-free guidance never evaluates the negative branch. **Every negative prompt in this pipeline is inert.** |
| `"the object floats free with no shadow beneath it"` | No effect. Diffusion models read the noun and drop the negation; naming "shadow" tends to summon one. |
| `"no ground, no horizon, no surface anywhere"` | Actively harmful — produced a floor and a horizon line every time. |
| Positive assertion: background is flat, two-dimensional, and continues *below* the object to the bottom edge | **Fixed the floor.** The horizon is gone. |
| LoRA weight 0.8 → 0.5 → none at all | Shadow unchanged in all three. It is the base model's prior, not the LoRA's. |

Four wordings and one controlled probe were enough to establish that the shadow is not
reachable from the prompt. So it is not argued with — the background is replaced.

## How it works

```bash
pip install rembg onnxruntime          # once; first run fetches u2net, ~170 MB

python postprocess_cat1.py --set-dir ../runs/2026-08-14_kino-sets-v9/noir --dry-run
python postprocess_cat1.py --set-dir ../runs/2026-08-14_kino-sets-v9/noir --sheet
```

For every category-1 card:

1. the object is cut out with u2net;
2. the background colour is sampled **from the generated image**, at the pixels the mask calls
   background, and a second tone of that same colour is derived from it;
3. the backdrop is drawn in code from the `bg_style` already recorded on the card — gradient,
   solid, stripes, rays or motif, the five kinds the brief allows;
4. the object is composited back with a slightly feathered edge.

Results go to `variant_N/final/`, next to `raw/`. Nothing is overwritten, and a
`postprocess_report.json` records the colours and mask coverage per card.

## Why colours are sampled and not looked up

The palette is stored as English names — "deep teal", "warm amber". Mapping those to RGB means
a lookup table that drifts out of sync with whatever the generator actually painted. Sampling
the real pixels means the redrawn card matches the rest of the set by construction.

Three details that each cost a debugging round:

- **Sample the original, not the cut-out.** rembg zeroes the RGB channels wherever it sets
  alpha to zero, so a cut-out's "background colour" is always black.
- **Four clusters, not two.** A striped backdrop carries three or more colours; with k=2 the
  orange and the teal average into a muddy olive belonging to neither.
- **The base colour is chosen by area and explicitly *not* by darkness.** The darkest cluster is
  usually the cast shadow this script exists to remove.

## One colour, two tones

A category-1 background is "one colour with a gradient, or a simple pattern". An earlier version
took the accent from the second cluster, picking the most saturated one so the set's amber
survived against its teal. That was the wrong reading of the brief: a two-hue backdrop competes
with the object instead of sitting behind it.

Now only the base is sampled. The second tone is derived from it — the same colour mixed towards
white if the base is dark, towards black if it is light — so stripes, motif, rays and gradient
all come out as one hue in two lightnesses. Deriving rather than sampling is what makes the rule
hold: even if the generator painted a second hue back there, it cannot reach the redrawn card.

The prompt templates in `set_designer.py` state the same rule in words. Both halves have to
agree; otherwise the generator paints one thing and the post-process replaces it with another.

## Scope

Only category 1 is touched. Categories 2-4 need their contact shadow — it is what seats the
object on its surface — and their backgrounds are meant to have depth.
