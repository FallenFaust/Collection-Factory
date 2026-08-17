# The tool

Everything else in this repository is a stage. This is the tool the brief asks for: a producer
gives it a theme and any wishes, and gets finished card sets back.

```
start.bat                    ← double-click
```

A page opens at `http://127.0.0.1:8720`. Type a theme, optionally some wishes, choose how many
variants and how many candidates per card, press the button. The page shows which stage is
running, how far along it is, and the finished cards as a gallery when it ends.

Before starting: ComfyUI running on port 8188. The API key is typed into the page — it pays
for inventing the set and for the judge, so it belongs to whoever is running the tool rather
than to whoever installed it. It is held for the length of the run and written nowhere: not to
the run folder, not to the log, not to a config file. Left empty, the run falls back to
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in the environment, which is how the command line keeps
working unattended.

Anthropic and OpenAI keys are both accepted; the provider is read from the key's prefix rather
than asked for in a dropdown.

## What one press does

```
theme + wishes
   │
   ├─ design      LLM invents ten objects per variant; category, rarity, pose,
   │              background and the final prompt are assembled in Python
   ├─ generate    ComfyUI renders N candidates per card
   ├─ judge       a vision model scores each candidate against its own card and picks one
   ├─ reframe     the object is measured and the crop zoomed until it fills the frame
   ├─ background  category-1 backgrounds are redrawn in code — no cast shadow, no floor
   └─ assemble    runs/<date>_<theme>/<set>/variant_N/final/01 Лупа детектива.png
```

`manifest.json` next to the variants carries the run's metrics: cards delivered, how many were
accepted on the first seed, how many went to a human, wall time, and the exact generation
settings used.

## Two entry points, one pipeline

| | for whom |
|---|---|
| `start.bat` → `pipeline/studio.py` | the producer: a form and a gallery |
| `pipeline/orchestrator.py --theme ...` | scripting, scheduled runs, CI |

Both call the same `produce()`. The page adds no logic of its own — it renders progress and
forwards a cancel switch, which is why a run behaves identically whichever way it is started.

## Stopping a run

The Stop button takes effect after the current image rather than at the end of the run;
everything already rendered stays on disk, and starting the same run again skips it. A
four-hour render should never be a commitment made by pressing a button once.

## Why no framework

`http.server` from the standard library and one inline page. Flask or Gradio would add an
install step that can fail on a machine that already has a 12 GB model, a LoRA and a working
ComfyUI on it — for a tool that serves one person on one machine. The whole interface is one
file with no dependencies, which is also why it will still run in a year.

## Not yet in the interface

Regenerating a single card from the gallery. The judge already returns a concrete prompt fix
for every rejection, and nothing consumes it yet; the natural home for that is a button under
the rejected card. See `docs/README_judge.md`.
