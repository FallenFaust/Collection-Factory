# The judge: scoring renders and choosing between them

The layer that makes this a pipeline rather than a batch script. Without it the generator's
output goes straight to a human; with it, the pipeline produces several candidates per card,
judges them against the card's own specification, and ships the best one.

## Why it exists, measured

A seed sweep on a frozen prompt — `runs/2026-08-15_seed-probe`, two objects across four
seeds — produced a correctly presented object on **three seeds out of four**. Same prompt,
same LoRA, same weight; only the seed differed.

That number is the whole argument:

- 25 % is far too high to ship blind;
- and it is not reachable by wording. Three rounds of prompt surgery were spent before the
  sweep was run, and two of them were chasing seed noise. The journal keeps them as negative
  results.

What remains is a selection problem, and selection is cheap: rendering three seeds costs
three times the GPU and removes almost all of the failures, provided something can tell a
good render from a bad one.

## Usage

```bash
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY

python judge.py --set-dir ../runs/2026-08-14_kino-sets-v12/noir --dry-run
python judge.py --set-dir ../runs/2026-08-14_kino-sets-v12 --sheet
```

For every card it collects `raw/<card_id>__s*.png`, scores each candidate, copies the best
passing one to `variant_N/selected/<card_id>.png`, and writes `judge_report.json` next to it.

## The rubric

Six axes, scored 1-5, where 3 means shippable:

| Axis | What it checks |
|---|---|
| `category` | the background matches the assigned category, no more and no less |
| `pose` | the object is presented in the pose the card asks for |
| `identity` | the object is the one named, recognisable by silhouette alone |
| `style` | casual painterly game art, not a photograph or a product render |
| `framing` | the object is large, centred, whole, nothing crops it |
| `cleanliness` | no text, no watermark, no second object, no stray props |

Plus four hard rejects that fail a card whatever the scores say: readable text as an object's
identity, banned content (weapons, ammunition, alcohol, tobacco), more than one subject, and
an object cropped by the frame.

The axes are the defects the journal actually recorded, in the order they cost the most work.
That is deliberate. A judge asked "is this a good image?" agrees with everything; a judge
asked "is the dial facing the camera, as this card requires?" disagrees usefully.

A card passes only if the model accepts it, no hard reject fired, **and** no single axis is
below 3. An image can average well and still be unusable on one axis — that is exactly the
compass, which scores full marks on style and identity while being unusable on pose.

## Failing closed

A verdict that will not parse is retried twice, and then the card is sent to review rather
than accepted. Failing open would silently ship unjudged images, which is the one outcome
worse than rejecting a good one.

## Metrics

`judge_metrics.json` at the root of the run:

- `first_try_pass_rate` — share of cards whose **first** seed was already shippable. This is
  the number that decides whether an extra seed is worth its GPU time.
- `generations_total` — candidates judged.
- `accepted` / `needs_review` — what shipped and what a human still has to look at.

These are the fields the top-level README promised and that read `metrics_pending` until now.

## The retry loop

A rejection is not the end of the line. `orchestrator.py` re-renders every rejected card, judges
it again, and only what still fails after three attempts goes to a human.

**The first two attempts change nothing but the seed.** That order is not laziness, it is what
the evidence says: the seed sweep found three seeds in four produce a correctly presented
object on an unchanged prompt, so most rejections are noise in the sampler rather than a fault
in the wording. Re-seeding also cannot damage the set — a rewritten prompt drifts away from the
nine cards around it, a new seed cannot.

**The third attempt rewrites the prompt.** Two bad seeds in a row means the defect is
systematic, and only then is it worth changing words. The rewrite is a separate model call that
receives the prompt, the defect and the reviewer's suggested fix, under two rules taken straight
from this project's own failures:

- never name the defect — "no cast shadow" produces a cast shadow, and "not tipped over"
  produces a tipped object;
- keep every other clause word for word, so the card stays in its set.

A rewrite that comes back much shorter or much longer than the original is discarded: it has
lost the category, background or style block along the way. A model that errors out leaves the
prompt untouched. The card simply spends that attempt on another seed instead.

The run's metrics gain `retried`, `recovered`, `retry_generations` and `amended`, and
`judge_report.json` keeps the full attempt history per card — seed, score, defect, and whether
the prompt had been rewritten by then. The prompt that finally passed is written back to
`cards.json`, so the record matches what shipped.
