# Collection Factory

A tool that produces artwork for collectible cards.

**In:** a set theme plus free-form wishes from the producer.
**Out:** 3–5 variants of the set, finished cards, and the run's metrics.

## Running it

```
start.bat
```

A double-click opens `http://127.0.0.1:8720`: theme, wishes, set composition, API key, one
button. From there the page shows which stage is running, how far along it is, and the
finished cards as a gallery. ComfyUI must be running on port 8188 first — see
`docs/README_studio.md` and `docs/COMFYUI_SETUP.md`.

The same run from the command line, for scripting and scheduled jobs:

```bash
python pipeline/orchestrator.py --theme "Movie collection: film noir" \
    --wishes "more brass and rain" --variants 3 --seeds-per-card 3
```

Both entry points call the same `produce()`. The page adds no logic of its own — it renders
progress and forwards a cancel switch — so a run behaves identically whichever way it starts.

```bash
pip install -r requirements.txt
```

---

## Architecture

```
   theme + producer's wishes
              │
   ┌──────────▼──────────┐
   │ 0. Style bible      │  style/style_bible.md + categories.json
   │    (offline, versioned) │  the reference cards, formalised
   └──────────┬──────────┘
              │
   ┌──────────▼──────────┐
   │ 1. Ideation (LLM)   │  N objects, laid out over a fixed slot plan
   │                     │  checked for IP, repeats, readability, age rating
   └──────────┬──────────┘
              │
   ┌──────────▼──────────┐
   │ 2. Prompt assembly  │  deterministic, from slots: object + pose +
   │                     │  framing + category + background + style
   └──────────┬──────────┘
              │
   ┌──────────▼──────────┐
   │ 3. Producer review  │  a pause before the GPU: edit the object, the
   │    (the page)       │  pose or the prompt; content edits recompile
   └──────────┬──────────┘
              │
   ┌──────────▼──────────┐
   │ 4. Generation       │  ComfyUI, N candidates per card
   └──────────┬──────────┘
              │
   ┌──────────▼──────────┐
   │ 5. Framing and      │  reframe.py — zoom by the object's mask
   │    category-1 bg    │  postprocess_cat1.py — background redrawn in code
   └──────────┬──────────┘  deterministic, before scoring, on every candidate
              │
   ┌──────────▼──────────┐
   │ 6. Judge (VLM)      │◄─┐ 6 axes + 4 hard rejects, picks one of N
   └──────────┬──────────┘  │
              │             │ 7. Retry: two attempts on a new seed,
              ├─────────────┘    the third rewrites the prompt. Limit 3.
              │ accept
   ┌──────────▼──────────┐
   │ 8. Assembly         │  final/ ships, review/ holds what a human must see
   └──────────┬──────────┘  manifest.json carries the run's metrics
              │
        3–5 variants of the set
```

**Stages 6 and 7 are what make this a pipeline rather than a batch script.** The judge scores
every candidate against the specification of its own card and picks one. A rejected card is
not the end of the line: the first two attempts change only the seed — a seed sweep on a
frozen prompt produced a correctly presented object on three seeds out of four, so most
rejections are sampler noise rather than a fault in the wording — and the third attempt
rewrites the prompt against the defect the judge named. Only what survives three attempts
goes to a person, and it goes to `review/`, never into the set.

---

## What is decided in code, and what is left to a model

This split is the main design decision in the project, and every stage repeats it.

**The model invents content only** — the object, the surface under it, the surrounding
environment, the card's name, and which of three poses the object needs.

**Everything else is assembled in Python** — the category of each slot, the background, the
rarity, the framing clause, the style block and the final prompt. The brief requires all four
object categories in every set; asked to distribute them itself, a language model drifts by
the third call, so the slot plan is fixed before the request and handed to the model as a
constraint.

The same split decides what happens with famous films. A set may quote one — up to four cards
of ten, switched on per run — but the quote is the *archetype* redrawn as our own object, and
the reference itself is metadata: it is recorded on the card and never sent to the image
generator. That is a legal line and a practical one at once. A prop design, an emblem and a
character likeness are owned; a battered brown fedora is not. And a generator given a franchise
name answers with a poster — lettering, a face, a collage — instead of one clean prop.
`docs/README_set_designer.md` has the rule and the checks that hold it.

The same principle covers the content rules. "No weapons, ammunition, alcohol or tobacco" is
stated in the ideation prompt **and** enforced by a regular expression over every object,
name, surface and environment. The second half matters more than the first: a hard
requirement is not left to a model's good word. When the check fires, the request is retried
with the reason attached, so the model corrects rather than rerolls.

---

## Layout

```
start.bat         launches the tool
style/            style_bible.md, categories.json — the source of truth on style
prompts/          ideation, judge and set-review prompts, versioned as markdown
pipeline/         studio.py (interface), orchestrator.py (the run),
                  set_designer.py (ideation and prompt assembly),
                  batch_generate.py + comfy_client.py (generation),
                  judge.py (scoring and the retry loop),
                  reframe.py, postprocess_cat1.py (deterministic repairs),
                  lora_grid_test.py (probes)
docs/             one README per module, with the reasoning behind it
runs/             runs: final/, review/, contact sheets, manifest.json
experiments.md    the journal: what was tried, what failed, why each choice was made
```

## Metrics

Every run writes `manifest.json`:

- `first_try_pass_rate` — share of cards accepted on their first seed
- `generations_total` — images spent on the set
- `retried` / `recovered` — how many cards entered the retry loop and how many it saved
- `retry_generations` / `amended` — what the loop cost: extra images and rewritten prompts
- `needs_review` — how many cards still went to a person
- `wall_time_sec` — time per run

This is what separates a producer's solution from an artist's one: the pipeline is measurable
and its cost is predictable.
