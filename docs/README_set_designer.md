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

The model's reply is validated before prompts are assembled, and the checks come in two
strengths (`validate_split`). The distinction was bought with a lost run, described at the
bottom of this section.

**Hard — the set is thrown away and the request retried** (`--retries`, default 2):

* exactly 10 cards, each with a name, an object and a pose from the closed list;
* category 3 cards have a surface, category 4 cards have an environment;
* the set palette is fully populated;
* no duplicate objects within a variant;
* no banned content — weapons, ammunition, alcohol, tobacco;
* no object whose identity is readable text;
* no franchise or brand name in any field that becomes a prompt — see below.

**Soft — the set is usable, so it is kept:** the variety limits below. They are sent back as
feedback and the model gets **one** more attempt; if the set still comes back monotonous it
ships with a warning in the log rather than being discarded.

**What this cost to learn.** A real "Film Horror" run was rejected three times and the variant
was dropped: `VHS` was read as a brand name (it is a format), `videotape cartridge` was read as
ammunition, and "6 of 10 objects are brass" — a matter of quality — killed the set outright.
Three rejections, three different reasons, nothing shipped. Two of them were the checks being
wrong and the third was the wrong severity. So:

* the franchise check now looks for **title case only**: `Millennium Falcon` is a franchise,
  `VHS` and `LP` and `CRT` are acronyms;
* `FALSE_ALARMS` excuses banned keywords inside phrases that make the innocent sense plain —
  `videotape cartridge`, `glue gun`, `fan blade`, `letter opener`, `wine-red velvet`. The rule
  for adding one: the phrase must be unambiguous on its own, which `cartridge` alone is not;
* variety became soft, and its material limit went from 3 to 4.

A check that fires on a correct set is worse than no check at all: it costs a retry, and the
retry makes the model change something that was not broken.

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

## Variety inside a set

A set can pass every other check and still be monotonous. A "Nautilus" run produced ten
distinct objects of which **ten were brass** and four were round-faced: no duplicate strings,
no banned content, nothing for the old checks to catch — and a set that read as one thing
photographed ten times. A comedy set gave a party hat and a top hat, a juggling ball and a
juggling pin.

The objects differed in name and not in kind, so the rule is stated in terms of kinds:

| Check | Limit |
|---|---|
| objects sharing a dominant material | 4 |
| objects with the same head noun (hat / hat) | 1 |
| round-faced objects — dials, mirrors, coins | 2 |
| any other word shared across objects | 2 |

The rule is written into the ideation prompt **and** checked here, for the same reason the
content policy is: a requirement that matters is not left to a model's good word. When a check
fires the request is repeated with the reason attached — "6 of 10 objects are brass, keep at
most 4" — so the model corrects a named fault instead of rerolling.

**These are soft checks.** Monotony is a quality problem, not a policy breach: the set gets one
corrective attempt and then ships with the warning in the log. Discarding a whole variant over
its palette of materials is a worse outcome than a slightly brassy set.

One implementation note worth keeping. The first version of the word splitter used
`w.strip(",.'s")`, which takes a *set* of characters: "brass" lost its final s and became
"bras", "glass" became "gla", and the material check silently matched nothing on the very set
that motivated it. Suffixes are stripped explicitly now.

## Franchise homages

A cinema collection that may not touch a single famous film is weaker than it needs to be:
recognition is half of what makes a card collectible. So the rule is not "no franchises" but
**no copies** — with `--homages` (a checkbox in the tool) up to four of the ten cards may quote
a film, and the quote is redrawn as our own object.

What "stylised" means, as the ideation prompt states it:

* quote the **archetype**, not the artefact — the kind of object a film made famous, designed
  again as ours: a battered brown felt fedora, not a screen-accurate replica of one;
* change at least one identifying feature: proportion, colour, material or ornament;
* drop everything that identifies the rights holder — logos, emblems, insignia, lettering,
  serial numbers, one character's house colours;
* no character likenesses, no vehicle or ship whose silhouette *is* the trademark;
* the Russian card name may allude — «Шляпа археолога» — but carries no trademarked title.

**The reference never reaches the image generator.** It is stored in the card's own `homage`
field, shown at the review step, written to `cards.json`, `prompts.csv` and the run manifest,
and left out of the prompt entirely. Two reasons point the same way. Legally, an archetype is
not owned and a specific prop design is. Technically, a franchise name makes Flux reproduce a
still: "Indiana Jones' hat" comes back as a poster — lettering, a face, a collage — while "a
battered brown felt fedora with a sweat-stained band" comes back as a card.

That is why `object`, `surface` and `environment` are checked for proper nouns. Any **title-case**
word that is not an era or a place (`ALLOWED_CAPS` — Victorian, Art Deco, Bakelite, Fresnel…)
fails validation with an instruction to move the name into `homage`. Title case, not any
capital: a franchise reads "Millennium Falcon", while an all-capital word in a prop description
is an acronym — VHS, LP, CRT — and rejecting a horror set over the letters VHS is exactly what
the first version did. The heuristic is
deliberately loud: a false positive costs one retry, a miss costs a prompt that asks the
generator to draw a trademark. The same check runs over the producer's own edits at the review
step, but there it only warns — at that point the producer is the authority.

Two things this stage cannot do, and where they are handled instead. A Russian card name
carrying a franchise title is not machine-checkable, so it is a prompt rule plus the review
step. And the generator may stamp a studio-looking emblem on an object nobody asked to brand —
that is a hard reject in the judge, which is the only stage that looks at the picture.

The content policy is not suspended for homages: a famous weapon is still a weapon.

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
| `--homages` | allow up to 4 stylised nods to famous films; off by default |
| `--offline` | run the pipeline on the built-in demo set without an API key |

`OPENAI_BASE_URL` overrides the endpoint, which is how any OpenAI-compatible provider gets
plugged in — useful for comparing models at this stage, as the brief asks for.

## Next

`prompts.csv` feeds the generation stage: a ComfyUI batch runner using the epoch and weight
chosen in `README_lora_test.md`, with several seeds per card. Auto-QA then rejects images whose
background does not match its assigned category, and assembles the final set variants.
