# LLM stage: theme → card set

The first stage of the pipeline. It takes a set theme and free-form wishes from the producer,
and returns 3-5 variants of the set, 10 cards each, with generation-ready prompts.

## Usage

```bash
export ANTHROPIC_API_KEY=...          # or OPENAI_API_KEY

python set_designer.py --theme "Movie collection: film noir" --wishes "more rain and neon"
python set_designer.py --theme "Movie collection: western" --variants 4
python set_designer.py --theme "Movie collection: horror" --offline   # no API, demo set
```

Output:

```
sets/<set_id>/
├── plan.json              # theme, wishes, and the list of variants with their concepts
├── variant_1/
│   ├── cards.json         # full data: palette, objects, categories, rarity, prompts
│   └── prompts.csv        # flat table for the batch generation stage
├── variant_2/
└── variant_3/
```

## Recompiling prompts after a template edit

```bash
python set_designer.py --rebuild ../runs/2026-08-14_kino-sets-v6
```

Rereads every `cards.json` under the path, rebuilds the prompts from the stored objects using
the current templates, and rewrites `prompts.csv`. No model calls, no cost, and — the point —
the objects stay identical, so the new render is directly comparable with the old one.

Without this, changing one clause in a template meant regenerating the whole set through the
LLM and getting different cards, which destroys the before/after comparison that makes a fix
verifiable.

Sets written before `surface` and `environment` became stored fields still carry them inside
the prompt text at fixed positions; `--rebuild` recovers them from there and saves them
properly, so the recovery only ever happens once.

## Why the category is assigned in code, not by the model

The brief requires **all four object categories to be present in every set** — that is what
holds a set together visually. A category is not defined by the object but by how it is
presented: floating on a gradient, standing against a flat wall, standing on a realistic
surface, or embedded in an environment.

Hand that choice to a language model and it drifts. The first call splits the categories
honestly; by the third, eight cards are "in an environment" because that is more fun to write.
So:

* **Python** fixes the slot plan `[1,1,2,2,3,3,3,4,4,4]` before the request and substitutes the
  background template for each category. The category distribution is guaranteed mechanically.
* **The LLM** receives the slot plan as a constraint and fills in content only: the object,
  the surface under it, the surrounding environment, and the card name.

A side benefit: background templates live in one place (`CATEGORY_TEMPLATES`) and apply
instantly to every set already generated — no repeat LLM calls needed to change them.

## Automatic checks

The model's reply is validated before prompts are assembled; on failure the request is retried
(`--retries`, default 2):

* exactly 10 cards;
* no duplicate objects within a variant;
* category 3 cards have a surface, category 4 cards have an environment;
* every card has a pose from the closed list;
* the set palette is fully populated;
* every card has a name.

The list of objects already used is passed forward between variants, so variant 3 does not come
back as a reshuffle of variant 1. That is what makes them "different readings of the theme"
rather than different seeds.

## Pose

Every card carries a `pose` — how the object is turned towards the viewer — chosen by the model
from three keys and compiled to fixed wording in code:

| Key | For | Wording |
|---|---|---|
| `front` | objects whose face carries their identity: compass, clock, mirror, medallion | seen straight from the front, its face turned fully towards the viewer and parallel to the picture plane |
| `upright` | objects with an obvious vertical axis: lamp, candlestick, bottle, helmet | standing upright and level, its vertical axis parallel to the side edges of the frame |
| `three_quarter` | handled or asymmetric objects: magnifying glass, telephone, camera, keys | in a gentle three-quarter view, its main axis leaning slightly from lower left to upper right |

The first attempt was one clause on every card demanding everything be upright and square to the
frame. It was wrong, and the render proved it: a magnifying glass at an angle reads perfectly —
lens to the viewer, handle down to the right — while a compass rolled the same way turns its dial
into an ellipse. The defect was never the tilt; it was that a compass has a face and the face was
not facing anyone. A single rule cannot express that, and applied blindly it would have spoiled
the cards that were already right.

Same split as the category slot plan: the model picks the key, the code owns the words. A closed
list means a pose can be validated on the way in, counted across a set, and reworded globally
without touching any set.

`--rebuild --repose` fills the field in for sets written before it existed: one request per
variant, objects untouched, so the re-render stays directly comparable with the previous one.
Without a key it falls back to a keyword list (`POSE_HINTS`), which is a stopgap and says so.

## Rarity

Assigned in code: 4 common, 3 uncommon, 2 rare, 1 epic, shuffled by seed (`--seed`). It does not
affect the artwork — per the brief, rarity does not change how an object is displayed. Gold cards
(category 5) are not generated: the assignment asks for sets without them.

## Language

Documentation, comments and console output are in English. **Card names and set titles are
generated in Russian on purpose** — they are product content for a Russian-language brief and a
Russian-language Miro board, not documentation.

## Options

| Flag | Meaning |
|---|---|
| `--variants N` | how many readings of the theme, 1-5 |
| `--provider` | `anthropic` or `openai`; defaults to whichever key is present |
| `--model` | specific model; also read from `CARDGEN_MODEL` |
| `--trigger` | LoRA trigger word, `LORACCG` by default |
| `--offline` | run the pipeline on the built-in demo set without an API key |

`OPENAI_BASE_URL` overrides the endpoint, which is how any OpenAI-compatible provider gets
plugged in — useful for comparing models at this stage, as the brief asks for.

## Next

`prompts.csv` feeds the generation stage: a ComfyUI batch runner using the epoch and weight
chosen in `README_lora_test.md`, with several seeds per card. Auto-QA then rejects images whose
background does not match its assigned category, and assembles the final set variants.
