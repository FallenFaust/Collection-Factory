#!/usr/bin/env python3
"""
set_designer.py — the LLM stage of the collectible card pipeline.

Input:  a set theme plus free-form wishes from the producer.
Output: 3-5 variants of the set, 10 cards each, with generation-ready prompts.

Separation of concerns (this is the important part):
  * the LLM invents CONTENT only — object, surface, environment, card name;
  * the category, background template, rarity and final prompt are assembled in Python.

Categories 1-4 from the brief are a hard requirement for every set, so they must not be
delegated to a language model: it will drift by the third call. The slot plan is fixed in
code before the request, and the model receives it as a constraint to fill in.

The API key is read from ANTHROPIC_API_KEY or OPENAI_API_KEY. Without a key, --offline
runs on a built-in demo set so the rest of the pipeline can be exercised end to end.

Note on language: card names are generated in Russian on purpose. They are product
content for a Russian-language brief, not documentation.

Examples:
    python set_designer.py --theme "Movie collection: film noir" --wishes "more rain and neon"
    python set_designer.py --theme "Movie collection: western" --variants 4
    python set_designer.py --theme "Movie collection: horror" --offline
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import re
import sys
import urllib.request
from pathlib import Path

TRIGGER = "LORACCG"

# --------------------------------------------------------------------------- #
# Object categories (from the brief, "general rules for choosing objects")
# --------------------------------------------------------------------------- #
CATEGORY_DOC = {
    1: "Object without small details, floating in the air. Background is a flat gradient or a simple pattern.",
    2: "Object without small details, standing on a surface. A simple vertical plane behind it.",
    3: "Object standing on a realistic surface. Background has volume, with a simple vertical plane behind.",
    4: "Object embedded in a realistic environment while remaining the central subject.",
}

# Framing. The first version of this clause said "the object remains the central subject" and
# the object came out at a quarter of the frame. It was replaced with "large, fills most of the
# frame, hero shot" and the object grew — to about half. Still small for a card that will be
# looked at as a thumbnail.
#
# Two changes here, both learned elsewhere in this file. The clause moved to the front, right
# behind the object and its pose, because a Flux prompt weights its opening far more than its
# tail — the same thing the pose probe showed. And the size stopped being an adjective: "large"
# is a comparison with nothing, while "reaching close to the top and bottom edges, only a narrow
# margin of background around it" describes a measurable picture.
#
# Even so, framing is not left to the prompt alone. `reframe.py` measures the object with the
# same segmentation mask the background rebuild uses and zooms until the coverage is right.
# The prompt gets the object most of the way there; the code guarantees the rest.
FRAMING = ("filling almost the whole frame, an extreme close-up hero shot, the object reaching "
           "close to the top and bottom edges of the image with only a narrow margin of "
           "background around it, the whole object visible and nothing cropping it")

# Pose. Three probes to get here, and the answer was never in the wording.
#
# The first fix was one clause on every card demanding everything be upright and square to the
# frame. The render killed it: a magnifying glass at an angle reads perfectly — lens to the
# viewer, handle down to the right — while a compass rolled the same way turns its dial into an
# ellipse. The defect was never the tilt. A compass has a face, and the face was facing nobody.
#
# So pose became per object, from a closed vocabulary — the model picks the key, the code owns
# the words, the same split that keeps the category slot plan honest. Four wordings of "front"
# were then tried on the compass: at the head of the prompt, at the tail, and twice over. All
# four came back tipped. A 4x3 grid — four wordings against LoRA 0.8 / 0.4 / none — finally
# separated the variables (`runs/2026-08-15_front-probe`):
#
#   * with NO LoRA every wording gives a dial facing the viewer. The tilt is Icon-3D-Flux's
#     prior, not a failure of language: the icons it was trained on lie back for volume.
#   * dropping the two words that asked the object to hover straightens it at every weight,
#     production 0.8 included. That phrase was the other half of the cause — it reads as
#     tumbling in mid-air, and the model obliges by tipping the object away from the camera.
#
# Hence the shape below: each pose is a prefix on the object plus a clause after it. "front
# view of X" leads, because camera language at the very start of a Flux prompt outweighs the
# same words buried further in. The flight phrase then moved out of the category-1 template and
# onto the pose — see POSE_FLOAT below for why it cannot be a blanket rule either way.
POSE_PREFIX = {
    "front": "front view of ",
    "upright": "front view of ",
    "three_quarter": "",
}
POSE_TEMPLATES = {
    # Objects with a face: dials, clocks, mirrors, medallions. The face is the whole point.
    "front": "the face turned to the camera, orthographic straight-on view",
    # Objects with a clear vertical axis: lamps, bottles, candlesticks, helmets.
    "upright": ("standing upright and level, its vertical axis parallel to the side edges of "
                "the frame"),
    # Handled or asymmetric objects, where a slight angle is the readable view.
    "three_quarter": ("in a gentle three-quarter view, the whole silhouette readable, its main "
                      "axis leaning slightly from lower left to upper right"),
}
# Whether category 1 asks the object to hover depends on the pose, and the four renders that
# established this form a complete 2x2:
#
#                     with "hovering"          without
#   compass  front    lies on its back         faces the camera
#   magnifier 3/4     reads perfectly          tips over backwards
#
# The phrase is neither good nor bad; it interacts with the pose. An object shown face-on reads
# "hovering" as tumbling and tips away from the camera. An object shown at three-quarters uses
# the same word to hold its axis, and loses it without. So the word is attached to the pose
# rather than to the category, and only category 1 — the only floating one — ever sees it.
POSE_FLOAT = {
    "front": "",
    "upright": "",
    "three_quarter": ", hovering weightlessly",
}


def pose_clause(pose: str, category: int) -> str:
    """The pose wording for one card, plus the float phrase where it belongs."""
    return POSE_TEMPLATES[pose] + (POSE_FLOAT[pose] if category == 1 else "")


DEFAULT_POSE = "upright"

# Used only when an older set is rebuilt and has no pose stored — see `guess_pose`.
POSE_HINTS = [
    ("front", r"compass|clock|watch|mirror|medallion|amulet|locket|pendant|coin|dial|badge|"
              r"emblem|crest|plate|record|disc|mask|barometer|gauge|shield"),
    ("three_quarter", r"magnifying glass|magnifier|telephone|handset|typewriter|camera|radio|"
                      r"gramophone|projector|suitcase|briefcase|satchel|chest|trunk|crate|box|"
                      r"book|folder|file|envelope|kettle|teapot|glove|boot|shoe|hat|fedora|"
                      r"keys|keyring|toolbox|microscope|telescope"),
]

# Style vocabulary read off the reference cards supplied with the brief. The first production
# run had none of this: composition and background were specified, style was left entirely to
# the LoRA, and the LoRA renders neutral product-grade 3D icons. Next to the references — thick
# painterly rendering, chunky exaggerated forms, saturated colour, bright key light — the output
# read as a different medium. Style has to be stated, not assumed.
# The palette belongs to the background. The first production run tinted the objects with
# it too — a desk lamp came out bright green on a green field, silhouette gone. Stated
# per template rather than once in STYLE, because it has to sit next to the background
# description it is correcting.
OBJ_COLOUR = ("the object keeps its own natural material colours and stays clearly "
              "separated from the background, never tinted to match it")

STYLE = ("casual mobile game art, thick painterly rendering, chunky rounded exaggerated shapes, "
         "saturated vivid colours, bright warm key light with soft coloured bounce light, "
         "glossy highlights, high contrast, clean readable silhouette, stylised not photorealistic")

# Category 1 gets the same style with the directional key light swapped for even ambient light.
# A cast shadow needs a light with a direction; remove the direction and the shadow has no
# reason to exist. This is the positive way to state it — see the note on the template below.
STYLE_FLOATING = STYLE.replace(
    "bright warm key light with soft coloured bounce light",
    "flat even ambient light wrapping the object equally from every side")

# The brief allows four kinds of background for category 1: a flat colour with a gradient, or
# a simple pattern — stripes, repeating shapes, or light rays. A single hard-coded gradient
# made every category 1 card look the same, so the choice is made per card.
#
# Every option is phrased to stay flat and lit from everywhere: a pattern must not turn into
# a scene, and nothing here may imply a surface the object could rest on.
#
# All five are MONOCHROME — one hue, two tones of it. The brief says «одноцветный с
# градиентом или простой узор»: a single colour, varying in lightness. Mixing the set's
# two palette colours here produced teal-and-amber stripes that fought the object for
# attention and read as a second subject rather than as a backdrop.
#
# The second tone is written as "a darker tone of it", never as "a deeper {bg_primary}":
# palette names already carry an adjective, so the latter compiles to "a deeper deep teal".
CAT1_BACKGROUNDS = {
    "gradient": ("one unbroken smooth gradient in {bg_primary} alone, darker at the bottom "
                 "and lighter towards the top, a single colour varying only in lightness"),
    "solid":    ("a completely flat field of solid {bg_primary}, one uniform colour "
                 "across the whole frame with no variation"),
    "stripes":  ("wide soft diagonal stripes in two tones of {bg_primary} alone, one lighter "
                 "and one darker, flat, evenly spaced, purely graphic"),
    "rays":     ("soft light rays fanning out symmetrically from behind the object, a lighter "
                 "tone of {bg_primary} over a darker tone of it, glowing and weightless"),
    "motif":    ("a flat decorative wallpaper pattern of small simple shapes in a lighter tone "
                 "of {bg_primary}, repeating evenly over a darker tone of the same colour"),
}
CAT1_BACKGROUND_ORDER = ["gradient", "rays", "stripes", "solid", "motif"]

# Category 2 gets the SAME five backgrounds as category 1, because the brief gives it the same
# list word for word: "одноцветный с градиентом, либо простой узор — полосы, повторяющиеся
# объекты, равномерный цвет или световые лучи". The categories differ in where the object is,
# not in what may be behind it. Sharing the dictionary rather than copying it means a change to
# the background vocabulary cannot apply to one category and miss the other.
#
# This replaces an invented option. Two probes had produced a spotlight cone on every category-2
# card (`runs/2026-08-15_cat2-probe`, `runs/2026-08-15_cat2-light-probe`); a flat wall, an
# explicitly full-width glow and non-directional lighting all failed to remove it, while vertical
# panels — which appear nowhere in the brief — worked. The reading was wrong. The cone was never
# a lighting problem: an undefined wall leaves the model nothing to render and it fills the gap
# with the one thing it knows about studio backdrops. Name a pattern from the brief's own list
# and there is no gap left to fill. Light rays are on that list, so the glow was never the
# defect either — a narrow shaft instead of a stated pattern was.
CAT2_BACKDROP_ORDER = CAT1_BACKGROUND_ORDER

CATEGORY_TEMPLATES = {
    # Third rewrite, and the reason is worth keeping. Version 7 said "no ground, no horizon,
    # no surface anywhere" and the renders came back with a floor and a horizon line every
    # time — the model does not read the negations, it reads the nouns, so we were effectively
    # asking for ground. Version 8 still had it.
    #
    # Everything here is now an assertion. The background is declared flat and two-dimensional
    # and, crucially, is declared to continue BELOW the object to the bottom edge: that leaves
    # no room in the frame for a floor to appear, without ever naming one. The words shadow,
    # ground, horizon and surface do not occur in this template at all.
    1: ("{trigger} {object}, {pose}, " + FRAMING + ", centered, isolated, "
        "nothing else in frame, "
        "the whole frame behind AND below the object is {background}, a completely flat "
        "two-dimensional field of colour like printed paper, continuing uninterrupted all the "
        "way down to the bottom edge of the image, " + OBJ_COLOUR + ", "
        + STYLE_FLOATING),
    2: ("{trigger} {object}, {pose}, " + FRAMING + ", standing on a plain flat surface, directly behind it "
        "{backdrop}, centered composition, clean simple shapes, no small details, "
        + OBJ_COLOUR + ", " + STYLE),
    3: ("{trigger} {object}, {pose}, " + FRAMING + ", standing on a realistic {surface}, behind it a plain vertical wall with a "
        "soft {bg_primary} to {bg_secondary} gradient, shallow depth, centered composition, "
        "gentle contact shadow, uncluttered background, " + OBJ_COLOUR + ", "
        + STYLE),
    # The environment is pushed out of focus on purpose. In the first production run the
    # backdrop phrase dominated the image: the prop shrank into a scene and the set lost its
    # shared palette, because "a rain-soaked city street at night" overrides any colour
    # instruction that follows it.
    4: ("{trigger} {object}, {pose}, " + FRAMING + ", in the foreground, close up, {environment} blurred behind it, "
        "shallow depth of field, the background heavily out of focus and secondary, "
        "{bg_primary} and {bg_secondary} background palette, warm cinematic lighting, " + OBJ_COLOUR + ", "
        + STYLE),
}

NEGATIVE = "text, watermark, logo, signature, ui, frame, border, collage, multiple objects, cropped object, blurry"

# NOTE: with Flux these negatives are currently INERT. The graph pins cfg to 1.0, which is
# correct for a guidance-distilled model, but classifier-free guidance is exactly what
# evaluates the negative branch — with CFG off, nothing reads this text. It is kept because
# the field is part of the data contract and a future graph (or another base model) may use
# it, but nothing here should be relied on. Anything that must not appear has to be handled
# positively in the prompt, or removed at the ideation stage.
CATEGORY_NEGATIVE = {
    1: "drop shadow, cast shadow, shadow on background, ground, floor, table, surface, reflection",
}

# Age-rating policy for the collection: no weapons, ammunition, alcohol or tobacco.
# Asked of the model in the system prompt AND enforced here, on the same principle as the
# category slot plan — a hard requirement does not get delegated to a language model's
# goodwill. A violation fails validation and the request is retried with the reason attached.
BANNED_CONTENT = re.compile(
    r"(?i)\b("
    r"gun|guns|pistol|revolver|rifle|shotgun|firearm|weapon|ammunition|ammo|"
    # "casing" alone is a false positive magnet: a compass has an ornate casing and a watch
    # has a brass one. Only the ammunition senses are banned.
    r"bullet|bullets|cartridge|bullet casing|shell casing|cartridge casing|spent casing|"
    r"magazine clip|holster|"
    r"knife|knives|dagger|switchblade|blade|sword|sabre|saber|machete|axe|hatchet|"
    r"grenade|bomb|explosive|brass knuckles|"
    r"whiskey|whisky|bourbon|scotch|rum|vodka|gin|brandy|cognac|liquor|liqueur|"
    r"wine|beer|ale|champagne|cocktail|martini|absinthe|moonshine|"
    r"decanter|hip flask|shot glass|wine glass|beer mug|tumbler|"
    r"cigarette|cigarettes|cigar|cigars|ashtray|tobacco|smoking pipe|rolling papers|"
    r"nicotine|vape|lighter and cigarettes"
    r")\b"
)

# Objects whose identity depends on legible text. Flux cannot render readable words, so a
# business card comes out embossed with "BUUN GEE DUNAI" — and a business card without a
# readable name has stopped being a business card. The fix belongs at ideation, not in the
# negative prompt: no amount of "no text" rescues an object that is made of text.
TEXT_DEPENDENT = re.compile(
    r"(?i)\b("
    r"business card|calling card|visiting card|name card|"
    r"newspaper|newsprint|magazine|tabloid|"
    r"letter|telegram|postcard|envelope with|"
    r"poster|billboard|banner|signboard|street sign|nameplate|"
    r"certificate|diploma|passport|licence plate|license plate|"
    r"ticket|receipt|invoice|contract|"
    r"open book|open ledger|price tag|label with"
    r")\b"
)

# Both lists above are keyword lists, and a keyword list fires on the wrong sense of a word.
# Measured on a real run: a horror set was rejected three times, and two of the three
# rejections were the checks being wrong — "a black plastic videotape cartridge" was read as
# ammunition, and «VHS» was read as a brand. A check that fails on correct sets is worse than
# no check: it costs a retry, and the model fixes the wrong thing next time.
#
# So the banned senses are excused where the phrase makes the sense plain. The rule for adding
# to this list: the phrase must be unambiguous on its own. "videotape cartridge" is; "cartridge"
# is not, which is why the word stays banned outside these phrases.
FALSE_ALARMS = re.compile(
    r"(?i)\b("
    r"videotape cartridge|video cartridge|tape cartridge|film cartridge|game cartridge|"
    r"ink cartridge|printer cartridge|toner cartridge|cassette cartridge|"
    r"glue gun|spray gun|nail gun|water gun|heat gun|"
    r"fan blade|propeller blade|turbine blade|saw blade|shoulder blade|blade of grass|"
    r"wine red|wine-red|wine coloured|wine colored|wine velvet|"
    r"letter opener|letter rack|letter seal|letter scale|"
    r"magazine rack|magazine holder"
    r")\b"
)


def excused(text: str, hit: re.Match) -> bool:
    """True when a banned keyword sits inside a phrase that makes its innocent sense plain."""
    return any(m.start() <= hit.start() and m.end() >= hit.end()
               for m in FALSE_ALARMS.finditer(text))

# --------------------------------------------------------------------------- #
# Franchise homages
# --------------------------------------------------------------------------- #
#
# A cinema collection that may not touch a single famous film is weaker than it needs to be:
# half of what makes a film prop collectible is that the player recognises it. The rule is
# therefore not "no franchises" but "no copies" — the set may quote a film, and the quote has
# to be redrawn as our own object.
#
# Two separate reasons, and they push the same way:
#
#   Legal — a specific prop design, a logo, an emblem, a costume and a character likeness
#   belong to their rights holder. An archetype does not: a bullwhip and a battered brown
#   fedora are not owned by anyone, a helmet with *that* silhouette is.
#
#   Technical — naming a franchise in the image prompt makes the generator try to reproduce
#   a still. Flux answers "Indiana Jones' hat" with a poster: lettering, a face, a montage,
#   and none of it is a single clean prop on a plain background. The same card asked for as
#   "a battered brown felt fedora with a sweat-stained band" comes back usable.
#
# So the franchise name is producer-facing metadata (`homage`, kept in cards.json and shown
# at the review step) and never enters the prompt. The `object` field stays a generic English
# noun phrase, which is what the whole prompt assembly already expects.
HOMAGE_LIMIT = 4

HOMAGE_RULE_ON = """- **Franchise homages are allowed, as stylised nods — never as copies.** Up to {limit} of
  the ten cards may quote a famous film; the rest carry the theme on their own. A homage is
  built like this:
  * quote the ARCHETYPE, not the artefact. Take the kind of object a film made famous and
    design our own version of it: a battered brown felt fedora, a bullwhip coiled on a
    leather belt, a dented silver hip-lamp — not a screen-accurate replica of the prop;
  * change at least one identifying feature — proportion, colour, material or ornament — so
    the object reads as ours;
  * drop everything that identifies the rights holder: logos, emblems, insignia, lettering,
    numbers, serial markings, house colours worn by one specific character;
  * no character likenesses, no faces, no vehicle or ship whose silhouette *is* the
    trademark;
  * **never name the film, the studio, the character or the actor in the `object`, `surface`
    or `environment` fields.** Those go to the image generator, and a generator given a
    franchise name reproduces a poster — lettering, a face, a collage — instead of one clean
    prop. Describe the object by its shape, material and one telling detail.
  * put the reference in the card's `homage` field, in English, for the producer's records:
    "Indiana Jones — the fedora". Leave it as an empty string for a card that is a pure
    archetype.
  * the Russian card name may allude — «Шляпа археолога» — but must not use a trademarked
    title or character name.
  The content policy above still applies to homages without exception: a famous weapon is
  still a weapon and does not enter the set."""

HOMAGE_RULE_OFF = """- **No franchise references.** No named films, studios, characters or actors, and no prop
  whose recognisability comes from one specific film. The set is built on genre archetypes:
  what the genre looks like, not what one picture in it looked like. Leave every `homage`
  field as an empty string."""

# Words that are capitalised in a perfectly generic English noun phrase. Everything else that
# comes back capitalised inside `object` is a proper noun — which, at this stage, means a
# franchise or a brand that has to move to the `homage` field before the prompt is compiled.
# This is a heuristic and it is deliberately a loud one: the cost of a false positive is one
# retry, the cost of a miss is a prompt that asks Flux to draw a trademark.
ALLOWED_CAPS = {
    "art", "deco", "nouveau", "victorian", "edwardian", "georgian", "regency", "gothic",
    "baroque", "rococo", "roman", "greek", "egyptian", "aztec", "mayan", "norse", "viking",
    "celtic", "japanese", "chinese", "korean", "indian", "persian", "turkish", "moroccan",
    "russian", "french", "italian", "spanish", "german", "dutch", "english", "british",
    "american", "mexican", "african", "arctic", "atlantic", "pacific", "earth", "mars",
    "moon", "polaroid", "bakelite", "fresnel", "morse", "geiger", "petri", "tesla",
    "faraday", "erlenmeyer", "phillips", "allen", "swiss", "west", "east", "north", "south",
}
# Title case only, on purpose. A franchise reads "Millennium Falcon", not "MILLENNIUM FALCON",
# while an all-capital word in a prop description is almost always a generic acronym: VHS, LP,
# CRT, UV, SOS. The first run with this check rejected a horror set over «VHS» — a format, not
# a brand. All-capital words are left to the review step and to the judge.
CAPITALISED = re.compile(r"\b([A-Z][a-z]{2,})")
TRADEMARK_MARKS = re.compile(r"[™®©]")


def naming_problems(index: int, card: dict) -> list[str]:
    """Catch a franchise name sitting in a field that is about to become an image prompt.

    `homage` is exempt: that field exists precisely to hold the name. Everything else here
    is compiled into the prompt verbatim.
    """
    problems = []
    for key in ("object", "surface", "environment"):
        text = str(card.get(key, "")).strip()
        if not text:
            continue
        if TRADEMARK_MARKS.search(text):
            problems.append(f"card {index}: a trademark mark in `{key}` — the field goes "
                            f"into the image prompt and must stay a plain description")
        for word in CAPITALISED.findall(text[1:] if text[:1].isupper() else text):
            if word.lower() in ALLOWED_CAPS:
                continue
            problems.append(
                f"card {index}: «{word}» in `{key}` looks like a franchise or brand name. "
                f"That field is sent to the image generator, which answers a franchise name "
                f"with a movie poster. Move the reference to `homage` and describe the "
                f"object itself — shape, material, one telling detail")
    return problems


# How many cards of each category a set contains. The brief's own sets are ten cards with all
# four categories present, which is the default; the composition is a parameter because a
# producer designing a collection has reason to weight it differently — a set built around
# environments wants more of category 4, a set of icons more of category 1.
#
# It stays a *plan fixed before the request*, not something the model decides. That is the
# same reason the categories were taken away from the LLM in the first place: asked to
# distribute them itself, it drifts towards whichever category is more fun to write.
DEFAULT_SLOTS = {1: 2, 2: 2, 3: 3, 4: 3}

# Rarity does not affect the artwork — it is metadata only. The mix keeps the 4:3:2:1 shape of
# a ten-card set at any size, so a twenty-card set is not suddenly all epic.
RARITY_MIX = [("common", 4), ("uncommon", 3), ("rare", 2), ("epic", 1)]


def build_slot_plan(slots: dict | None = None) -> list[int]:
    """The category of every card position, in order."""
    slots = slots or DEFAULT_SLOTS
    plan = []
    for cat in (1, 2, 3, 4):
        plan += [cat] * max(0, int(slots.get(cat, slots.get(str(cat), 0))))
    return plan


def build_rarity_plan(n: int) -> list[str]:
    """`n` rarities in the 4:3:2:1 proportion, rounded so the list is exactly `n` long."""
    total = sum(w for _, w in RARITY_MIX)
    out = []
    for name, weight in RARITY_MIX:
        out += [name] * round(n * weight / total)
    # Rounding can land a card short or over; the commonest rarity absorbs the difference.
    while len(out) < n:
        out.append(RARITY_MIX[0][0])
    return out[:n]


SLOT_PLAN = build_slot_plan()


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #
def detect_provider(key: str) -> str:
    """Which API a key belongs to, by its prefix. A producer should not have to also pick the
    provider from a dropdown when the key already says which one it is."""
    key = (key or "").strip()
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("sk-"):
        return "openai"
    return ""


class LLM:
    """Thin SDK-free client: Anthropic, or any OpenAI-compatible endpoint."""

    def __init__(self, provider: str = "", model: str = "", timeout: int = 120,
                 key: str = ""):
        """`key` wins over the environment. The tool is used by whoever launched it, and their
        key should not be an environment variable someone else set up on this machine."""
        self.timeout = timeout
        if not provider:
            provider = detect_provider(key) or (
                "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai")
        self.provider = provider
        if provider == "anthropic":
            self.key = key or os.environ.get("ANTHROPIC_API_KEY", "")
            self.model = model or os.environ.get("CARDGEN_MODEL", "claude-sonnet-4-5")
            self.url = "https://api.anthropic.com/v1/messages"
        else:
            self.key = key or os.environ.get("OPENAI_API_KEY", "")
            self.model = model or os.environ.get("CARDGEN_MODEL", "gpt-4o")
            self.url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions"
        if not self.key:
            sys.exit(
                f"No API key for provider '{provider}'.\n"
                "Set ANTHROPIC_API_KEY or OPENAI_API_KEY, or run with --offline."
            )
        # A key goes into an HTTP header, and headers are latin-1. Anything else fails deep
        # inside urllib with "'latin-1' codec can't encode characters in position 0-4", which
        # says nothing about the actual mistake — a placeholder left in the environment
        # variable instead of the key.
        if not self.key.isascii():
            sys.exit(
                f"The API key in the environment is not a key: it contains non-ASCII "
                f"characters ({self.key[:12]}...). Looks like a placeholder was left in place "
                f"of the real value."
            )

    def ask(self, system: str, user: str) -> str:
        if self.provider == "anthropic":
            body = {
                "model": self.model,
                "max_tokens": 4000,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            headers = {
                "content-type": "application/json",
                "x-api-key": self.key,
                "anthropic-version": "2023-06-01",
            }
        else:
            body = {
                "model": self.model,
                "max_tokens": 4000,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            headers = {"content-type": "application/json", "authorization": f"Bearer {self.key}"}

        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode())
        if self.provider == "anthropic":
            return "".join(b.get("text", "") for b in data.get("content", []))
        return data["choices"][0]["message"]["content"]


def extract_json(raw: str):
    """Pull JSON out of a model reply, even when wrapped in ```json ... ```."""
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in the model reply:\n{raw[:400]}")
    return json.loads(raw[start:end + 1])


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #
SYSTEM = """You are an art director for collectible cards in a casual mobile game.
Art style: casual mobile 3D render, soft rounded shapes, warm light, readable silhouette,
low detail density — think Gardenscapes or Township.

You invent the CONTENT of the cards. You do NOT decide composition, background or category —
the pipeline assigns those. Your job is to pick an object that sits well in the given category.

Rules:
- Exactly one object per card. No characters, no people.
- **Every object must be a portable prop — something a person could pick up and carry.**
  Never architecture, buildings, fixtures bolted to a street or wall, staircases, doorways,
  vehicles, or any part of the environment itself. A street lamp, a phone box, a fire escape
  and a neon sign are all WRONG. A lantern, a telephone handset, a folded map and a brass
  key are right. This holds for category 4 as well: there the prop is *placed in* a setting,
  it is not the setting.
- **A set is ten different KINDS of thing, not ten names for one thing.** This is the rule
  that gets broken most often, and it is not about repeated words. A "Nautilus" set came back
  as a brass compass, a brass barometer, a brass depth gauge, a brass sextant, a brass bell,
  a brass lantern and a brass telescope: ten distinct names, one material, three of them the
  same round dial. Nothing in it repeated, and the set was still monotonous. So, within one
  set:
  * **materials must vary** — no more than four objects sharing a dominant material. Mix
    brass with wood, glass, fabric, ceramic, leather, paper, stone, enamel;
  * **silhouettes must vary** — no more than two round-faced objects, no two objects of the
    same family (not a juggling ball and a juggling pin, not a party hat and a top hat, not
    a jester's staff and a jester's collar);
  * **scale must vary** — something that fits in a palm, something held in two hands,
    something the size of a suitcase.
- **The producer's wishes set the mood, not a single material or motif.** "Nautilus" means
  deep-sea Victorian adventure — brass instruments *and* a diving suit's canvas glove, a
  coral specimen, a hand-drawn chart, a cork-stoppered vial. Taking one word literally and
  applying it to all ten cards is the failure mode, not the goal.
- **Give each object one distinguishing detail**, not a bare noun. "A brass compass" is a
  placeholder; "a brass compass with a cracked enamel dial" is a card. Keep it to a short
  noun phrase, but let that phrase carry a material, a colour or one telling feature.
- The object must be instantly recognisable by its silhouette.
- For categories 1 and 2 the object must be simple, without small details. This is about the
  object's geometry, not about the description: a simple shape can still be a specific thing.
- The object description is in English: a short noun phrase, with no background or lighting.
- The card name is in RUSSIAN, 1-3 words, lively rather than bureaucratic.
- **The name must not claim attributes the object description does not state.** If the name
  says the hat is black, the English description must say "a black fedora hat". Otherwise
  the picture and the caption will disagree.
- **Content policy — these are forbidden outright, in every genre:**
  weapons of any kind (guns, revolvers, rifles, knives, daggers, swords, axes, brass
  knuckles); ammunition, bullets, cartridges and spent casings; alcohol and its vessels
  (whiskey, wine, beer, cocktails, decanters, hip flasks, bar glasses); cigarettes, cigars,
  smoking pipes, ashtrays and tobacco.
  A genre is carried by its harmless props. Noir works through a magnifying glass, a rotary
  telephone, a case file, a fedora, a desk lamp. Horror works through a candelabrum, a music
  box, a pocket watch, an old key. Do not smuggle a banned item in by describing it
  indirectly — "spent brass cylinder" is still a casing.
- **No object whose identity depends on readable text.** The generator cannot render words:
  a business card, a newspaper, a poster, a telegram or a licence plate all come out covered
  in gibberish, and a business card nobody can read has stopped being a business card. Pick
  props that are recognised by their shape — a magnifying glass, a pocket watch, a bell.
- **Every card needs a pose — how the object is turned towards the viewer.** Pick exactly
  one of three keys, by the shape of the object, not by taste:
  `front` — the object has a face that carries its identity and must look at the viewer:
    a compass, a clock, a pocket watch, a mirror, a medallion, a barometer. Shown at an
    angle, the dial becomes an ellipse and the card stops reading.
  `upright` — the object has an obvious vertical axis and stands on it: a desk lamp, a
    candlestick, a bottle, a helmet, a bell, a vase.
  `three_quarter` — the object is handled or asymmetric and a slight angle is its most
    readable view: a magnifying glass, a rotary telephone, a camera, a suitcase, a folder,
    a bunch of keys.
- **The palette must stay in the game's bright, saturated register, whatever the genre is.**
  Mood comes from the objects and the lighting, never from draining the colour out of the
  frame. A horror or noir set still uses rich saturated colour — deep teal, plum, warm amber,
  bottle green. Never charcoal, grey, ash, muted, washed-out, desaturated or near-black:
  those turn a collectible card into a photograph and break the collection's visual register.

Reply with valid JSON only, no commentary."""

USER_TMPL = """Set theme: {theme}
Producer wishes: {wishes}
This is variant {variant} of {total}. It must be an independent interpretation of the theme,
not a reshuffle of the previous ones.
{avoid}

{homage_rule}

Build a set of exactly 10 cards following this slot plan (keep the order):
{slots}

Categories:
1 — object floats in the air on a gradient background. Pick a very simple object.
2 — object stands on a surface with a flat wall behind. Also a simple object.
3 — object stands on a realistic surface. Fill in "surface" — what it stands on.
4 — a portable prop standing in a realistic setting. Fill in "environment" — a SHORT backdrop
    phrase with "a"/"an", three or four words at most. It is scenery behind the prop, not the
    subject of the card, so do not describe a whole scene there.

Return JSON:
{{
  "set_title": "set name in Russian, 1-3 words",
  "concept": "one sentence IN RUSSIAN: how this variant differs from other readings of the theme",
  "palette": {{
    "bg_primary": "main background colour in English — SATURATED and reasonably bright",
    "bg_secondary": "second gradient colour in English — also saturated, must contrast with the first",
    "note": "no pattern field: the category-2 backdrop is chosen in code, not invented here"
  }},
  "cards": [
    {{
      "slot": 1,
      "name": "card name in Russian",
      "object": "short english noun phrase — no film, studio, character or actor names",
      "homage": "the film this card nods to, in English, or an empty string",
      "pose": "front | upright | three_quarter — see the pose rule",
      "surface": "category 3 only, empty string otherwise",
      "environment": "category 4 only, with an article, empty string otherwise"
    }}
  ]
}}"""


def ask_variant(llm: LLM, theme: str, wishes: str, variant: int, total: int,
                used: list[str], problems: list[str] | None = None,
                plan: list[int] | None = None, homages: bool = False) -> dict:
    slots = "\n".join(
        f"  card {i+1}: category {c} — {CATEGORY_DOC[c]}"
        for i, c in enumerate(plan or SLOT_PLAN)
    )
    avoid = ""
    if used:
        avoid = "These objects are already taken by other variants; avoid them and close analogues:\n" + \
                ", ".join(used)
    user = USER_TMPL.format(
        theme=theme, wishes=wishes or "no particular wishes",
        variant=variant, total=total, slots=slots, avoid=avoid,
        homage_rule=(HOMAGE_RULE_ON.format(limit=HOMAGE_LIMIT) if homages else HOMAGE_RULE_OFF),
    )
    # A retry that simply re-rolls the same request tends to repeat the same mistake.
    # Telling the model what failed turns the retry into a correction.
    if problems:
        user += ("\n\nYour previous attempt was rejected for these reasons. Fix all of them:\n- "
                 + "\n- ".join(problems))
    return extract_json(llm.ask(SYSTEM, user))


# --------------------------------------------------------------------------- #
# Validation and prompt assembly
# --------------------------------------------------------------------------- #
def validate(data: dict, plan: list[int] | None = None,
             homages: bool = False) -> list[str]:
    """Every problem in one list. `validate_split` is what the pipeline uses."""
    hard, soft = validate_split(data, plan, homages)
    return hard + soft


def validate_split(data: dict, plan: list[int] | None = None,
                   homages: bool = False) -> tuple[list[str], list[str]]:
    """Problems, separated into the ones that must fail a set and the ones that only warn.

    The distinction cost a real run to learn. A horror set was rejected three times and
    skipped entirely: once for a check that was wrong, once for a check that was wrong, and
    once for "6 of 10 objects are brass" — which is a matter of quality, not of policy.
    Losing the whole variant over it is the wrong trade.

    **Hard** — the set is unusable or breaks a promise made to the client: banned content, an
    object made of text, a franchise name heading for the image prompt, a missing field, the
    wrong number of cards. These fail the set.

    **Soft** — the set would ship, but it could be better: monotonous materials, repeated
    silhouettes, one motif circled. These are sent back as feedback on the next attempt, and
    if the model still will not fix them, the set is used with a warning in the log.
    """
    hard: list[str] = []
    soft: list[str] = []
    cards = data.get("cards")
    if not isinstance(cards, list):
        return ["no cards list"], []
    problems = hard
    plan = plan or SLOT_PLAN
    if len(cards) != len(plan):
        problems.append(f"got {len(cards)} cards, expected {len(plan)}")
    pal = data.get("palette") or {}
    for k in ("bg_primary", "bg_secondary"):
        if not str(pal.get(k, "")).strip():
            problems.append(f"empty palette.{k}")
    seen = set()
    for i, c in enumerate(cards[: len(plan)]):
        cat = plan[i]
        obj = str(c.get("object", "")).strip().lower()
        if not obj:
            problems.append(f"card {i+1}: empty object")
        elif obj in seen:
            problems.append(f"card {i+1}: duplicate object '{obj}'")
        seen.add(obj)
        if not str(c.get("name", "")).strip():
            problems.append(f"card {i+1}: missing name")
        blob = (f"{obj} {c.get('name', '')} "
                f"{c.get('surface', '')} {c.get('environment', '')}")
        txt = TEXT_DEPENDENT.search(obj)
        if txt and not excused(obj, txt):
            problems.append(
                f"card {i+1}: '{txt.group(0)}' in '{obj}' is an object made of text — "
                f"the generator cannot render readable words, so it would come out as "
                f"gibberish. Replace it with a prop that reads by its shape")
        hit = next((m for m in BANNED_CONTENT.finditer(blob) if not excused(blob, m)), None)
        if hit:
            problems.append(
                f"card {i+1}: forbidden content '{hit.group(0)}' in '{obj}' — "
                f"weapons, ammunition, alcohol and tobacco are banned in every genre; "
                f"replace it with a harmless prop that carries the same mood")
        pose = str(c.get("pose", "")).strip()
        if pose not in POSE_TEMPLATES:
            problems.append(
                f"card {i+1}: pose '{pose or 'missing'}' is not one of "
                f"{', '.join(POSE_TEMPLATES)} — pick the one that matches the object's shape")
        if cat == 3 and not str(c.get("surface", "")).strip():
            problems.append(f"card {i+1}: category 3 without a surface")
        if cat == 4 and not str(c.get("environment", "")).strip():
            problems.append(f"card {i+1}: category 4 without an environment")
        problems += naming_problems(i + 1, c)

    # The homage budget is a set-level property, so it is counted here rather than asked of
    # the model card by card. Ten quotes in a row is a film quiz; a few is a collection with
    # something to recognise in it.
    quoted = [str(c.get("homage", "")).strip() for c in cards[: len(plan)]]
    quoted = [q for q in quoted if q]
    if not homages and quoted:
        problems.append(f"{len(quoted)} cards carry a franchise homage, and this set was "
                        f"ordered without them — rebuild those cards as genre archetypes")
    elif len(quoted) > HOMAGE_LIMIT:
        problems.append(f"{len(quoted)} of the cards are franchise homages, at most "
                        f"{HOMAGE_LIMIT} may be — the rest must carry the theme on their own")

    soft += variety_problems([str(c.get("object", "")).strip()
                              for c in cards if str(c.get("object", "")).strip()])
    return hard, soft


def guess_pose(obj: str) -> str:
    """Fallback pose for a card written before poses existed, from a keyword list.

    Only ever used by --rebuild without a key. It is a stopgap, not the mechanism: the
    vocabulary is small enough to guess a compass right and wide enough to guess something
    unlisted wrong, which is why --repose asks the model instead.
    """
    low = obj.lower()
    for key, pattern in POSE_HINTS:
        if re.search(pattern, low):
            return key
    return DEFAULT_POSE


POSE_SYSTEM = """You are an art director for collectible cards. You assign each object a pose:
how it is turned towards the viewer. Reply with valid JSON only, no commentary."""

POSE_USER = """Assign a pose to each object below. Choose strictly from three keys:

front — the object has a face that carries its identity and must look at the viewer:
  a compass, a clock, a pocket watch, a mirror, a medallion, a barometer. At an angle the
  dial becomes an ellipse and the card stops reading.
upright — the object has an obvious vertical axis and stands on it: a desk lamp, a
  candlestick, a bottle, a helmet, a bell, a vase.
three_quarter — the object is handled or asymmetric and a slight angle is its most readable
  view: a magnifying glass, a rotary telephone, a camera, a suitcase, a folder, keys.

Objects, in order:
{objects}

Return JSON: {{"poses": ["front", "upright", ...]}} — exactly {n} values, same order."""


def ask_poses(llm: LLM, objects: list[str]) -> list[str]:
    """Poses for objects that already exist. Content is not touched, so a set repose-d this
    way stays directly comparable with its previous render."""
    listing = "\n".join(f"{i+1}. {o}" for i, o in enumerate(objects))
    data = extract_json(llm.ask(POSE_SYSTEM,
                                POSE_USER.format(objects=listing, n=len(objects))))
    poses = data.get("poses") or []
    if len(poses) != len(objects):
        raise ValueError(f"model returned {len(poses)} poses for {len(objects)} objects")
    bad = [p for p in poses if p not in POSE_TEMPLATES]
    if bad:
        raise ValueError(f"unknown pose(s): {', '.join(sorted(set(bad)))}")
    return poses


# --------------------------------------------------------------------------- #
# Variety inside one set
# --------------------------------------------------------------------------- #
#
# A set can pass every other check and still be monotonous. The "Nautilus" run produced ten
# distinct objects of which nine were brass and three were round dials: no duplicates, no
# banned content, nothing to catch — and a set that reads as one thing photographed ten times.
#
# The rule lives in the ideation prompt too, but it lives here as well for the same reason the
# content policy does: a requirement that matters is not left to a model's good word. When a
# check fires the request is retried with the reason attached, so the model corrects a named
# fault rather than rolling the dice again.
MATERIALS = ("brass", "bronze", "copper", "silver", "gold", "golden", "steel", "iron",
             "chrome", "wooden", "wood", "oak", "mahogany", "leather", "velvet", "silk",
             "porcelain", "ceramic", "glass", "crystal", "paper", "stone", "marble",
             "enamel", "plastic", "canvas", "wicker", "tin", "pewter")

# Objects whose identity is a round face. More than two of them and a set turns into a wall
# of dials — the exact defect the Nautilus run had.
ROUND_FACED = ("compass", "barometer", "gauge", "clock", "watch", "dial", "mirror",
               "medallion", "locket", "coin", "plate", "disc", "record", "porthole")

# Words too common to mean anything when shared between two objects.
NEUTRAL = {"a", "an", "the", "with", "and", "of", "small", "large", "old", "antique",
           "vintage", "ornate", "polished", "bright", "dark", "light", "round", "tall",
           "red", "blue", "green", "yellow", "black", "white", "brown", "grey", "gray",
           "orange", "purple", "pink", "teal", "amber", "handles", "handle"}

# Four of ten, not three. Three is a defensible target and a bad threshold: a set with four
# brass objects is a themed set, and rejecting it costs a retry that usually breaks something
# else. These are soft limits — see `validate_split` — so the number decides when the model
# gets told to try again, not when a variant is thrown away.
MAX_SAME_MATERIAL = 4
MAX_ROUND_FACED = 2


def clean_words(obj: str) -> list[str]:
    """Words of a noun phrase, punctuation and plurals removed.

    `strip(",.'s")` was the first attempt and it is a trap: strip takes a SET of characters,
    so "brass" lost its final s and became "bras", "glass" became "gla", and the material
    check silently matched nothing. Suffixes are stripped explicitly instead.
    """
    out = []
    for raw in re.split(r"[\s,]+", obj.lower()):
        w = raw.strip(".,;:!?\"'()")
        if w.endswith("'s") or w.endswith("\u2019s"):
            w = w[:-2]
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]                      # ribbons -> ribbon, keeps brass and glass intact
        if w:
            out.append(w)
    return out


# A trailing clause hides the head noun: "a brass telescope with a leather grip" ends on
# "grip", which is not what the card is of. Cutting at the first preposition puts the head
# back where it belongs.
TAIL = re.compile(r"\b(with|in|of|on|under|inside|holding|wrapped|tied|filled)\b")


def head_noun(obj: str) -> str:
    """The head of a noun phrase — "a brass telescope with a leather grip" -> "telescope"."""
    words = clean_words(TAIL.split(obj.lower())[0])
    for w in reversed(words):
        if w not in NEUTRAL and len(w) > 2:
            return w
    return words[-1] if words else ""


def variety_problems(objects: list[str]) -> list[str]:
    """What makes this set monotonous, in words the model can act on."""
    problems = []
    if len(objects) < 3:
        return problems
    words = [clean_words(o) for o in objects]

    counts = collections.Counter(m for ws in words for m in set(ws) if m in MATERIALS)
    for material, n in counts.items():
        if n > MAX_SAME_MATERIAL:
            problems.append(
                f"{n} of {len(objects)} objects are {material} — a set needs varied materials; "
                f"keep at most {MAX_SAME_MATERIAL} and replace the rest with wood, glass, "
                f"fabric, ceramic, leather or paper")

    heads = collections.Counter(head_noun(o) for o in objects)
    for noun, n in heads.items():
        if n > 1 and noun:
            same = [o for o in objects if head_noun(o) == noun]
            problems.append(f"{n} objects are the same kind of thing ({noun}): "
                            f"{', '.join(same)} — replace all but one")

    dials = [o for o in objects if any(r in o.lower() for r in ROUND_FACED)]
    if len(dials) > MAX_ROUND_FACED:
        problems.append(f"{len(dials)} objects are round-faced ({', '.join(dials)}) — "
                        f"keep at most {MAX_ROUND_FACED}, the rest of the set needs other "
                        f"silhouettes")

    shared = collections.Counter(w for ws in words for w in set(ws)
                                 if w not in NEUTRAL and w not in MATERIALS and len(w) > 3)
    for word, n in shared.items():
        if n >= 3:
            problems.append(f"the word «{word}» appears in {n} objects — the set is circling "
                            f"one motif instead of covering the theme")
    return problems


def cat1_background(pal: dict, key: str) -> str:
    """Resolve one of the brief's category-1 background kinds against the set palette."""
    frag = CAT1_BACKGROUNDS.get(key, CAT1_BACKGROUNDS["gradient"])
    return frag.format(bg_primary=pal["bg_primary"], bg_secondary=pal["bg_secondary"])


def cat2_backdrop(pal: dict, key: str) -> str:
    """The same background kind as category 1, worded as the wall standing behind the object."""
    return "a plain vertical wall showing " + cat1_background(pal, key)


def pick_cat1_backgrounds(rng: random.Random, count: int) -> list[str]:
    """Distinct background kinds for the category-1 cards of one set, so the two of them do
    not come out as the same picture twice."""
    order = CAT1_BACKGROUND_ORDER[:]
    rng.shuffle(order)
    return [order[i % len(order)] for i in range(count)]


def build_cards(data: dict, set_id: str, variant: int, trigger: str, seed: int,
                plan: list[int] | None = None) -> list[dict]:
    pal = data["palette"]
    rng = random.Random(seed)
    plan = plan or SLOT_PLAN
    rarities = build_rarity_plan(len(plan))
    rng.shuffle(rarities)
    cat1_keys = pick_cat1_backgrounds(rng, plan.count(1))
    cat1_iter = iter(cat1_keys)
    # The two category-2 cards of a set should not be the same wall twice either.
    cat2_iter = iter(pick_cat1_backgrounds(rng, plan.count(2)))

    out = []
    for i, raw in enumerate(data["cards"][: len(plan)]):
        cat = plan[i]
        bg_style = next(cat1_iter) if cat == 1 else (next(cat2_iter) if cat == 2 else "")
        # A pose is required of the model and validated, so this fallback should never fire;
        # it exists so --offline and any future caller cannot produce a card without one.
        pose = str(raw.get("pose", "")).strip()
        if pose not in POSE_TEMPLATES:
            pose = guess_pose(str(raw["object"]))
        prompt = CATEGORY_TEMPLATES[cat].format(
            trigger=trigger,
            object=POSE_PREFIX[pose] + str(raw["object"]).strip().rstrip("."),
            pose=pose_clause(pose, cat),
            surface=str(raw.get("surface", "")).strip() or "wooden surface",
            environment=str(raw.get("environment", "")).strip() or "a themed environment",
            background=cat1_background(pal, bg_style) if cat == 1 else "",
            backdrop=cat2_backdrop(pal, bg_style) if cat == 2 else "",
            bg_primary=pal["bg_primary"],
            bg_secondary=pal["bg_secondary"],
        )
        negative = ", ".join(x for x in (NEGATIVE, CATEGORY_NEGATIVE.get(cat, "")) if x)
        out.append({
            "card_id": f"{set_id}_v{variant}_{i+1:02d}",
            "n": i + 1,
            "name": str(raw["name"]).strip(),
            "object": str(raw["object"]).strip(),
            # Producer-facing only: what the card nods to. Deliberately absent from the
            # prompt — see the homage block above.
            "homage": str(raw.get("homage", "")).strip(),
            # kept so prompts can be recompiled later without asking the model again
            "surface": str(raw.get("surface", "")).strip(),
            "environment": str(raw.get("environment", "")).strip(),
            "bg_style": bg_style,
            "pose": pose,
            "category": cat,
            "rarity": rarities[i],
            "prompt": " ".join(prompt.split()),
            "negative": negative,
        })
    return out


# --------------------------------------------------------------------------- #
# Recompiling prompts without calling the model again
# --------------------------------------------------------------------------- #
#
# Template edits used to mean regenerating the whole set through the LLM, which changed the
# objects and destroyed comparability with the previous run — a heavy price for adding one
# clause to a prompt. The content and the presentation are separate concerns, so re-rendering
# the second from stored copies of the first should be free. --rebuild does exactly that.

# Sets written before `surface` and `environment` were stored as fields still carry them
# inside the prompt text, at fixed positions in the template. Recover rather than refuse.
SURFACE_RE = re.compile(r"standing on a realistic (.+?), behind it a plain vertical wall")
ENVIRONMENT_RE = re.compile(r"in the foreground, close up, (.+?) blurred behind it")


def rebuild_variant(cards_json: Path, llm: LLM | None = None,
                    repose: bool = False) -> tuple[int, list[str], int]:
    """Recompile every prompt in one variant from its stored content.

    Returns (cards, notes, model_calls). The call count is returned rather than assumed: an
    earlier version printed "one call per variant" whenever a key was present, and quietly
    made none at all, because every card already carried a pose and the request was skipped.
    A report that cannot be wrong about this is worth three lines.
    """
    payload = json.loads(cards_json.read_text(encoding="utf-8"))
    pal, notes, calls = payload["palette"], [], 0

    # Poses arrived after these sets were written. One request per variant fills them in for
    # the objects already stored — the content is untouched, so the re-render stays comparable.
    # --repose asks again for every card, not only the ones missing a pose: the fallback
    # keyword list defaults two cards in three to "upright", and its guesses are exactly what
    # a re-pose is meant to replace.
    targets = (list(payload["cards"]) if repose
               else [c for c in payload["cards"] if c.get("pose") not in POSE_TEMPLATES])
    if targets and llm is not None:
        try:
            for card, pose in zip(targets, ask_poses(llm, [c["object"] for c in targets])):
                card["pose"] = pose
            calls = 1
        except Exception as e:
            notes.append(f"{cards_json.parent.name}: could not ask the model for poses ({e}), "
                         f"falling back to the keyword list")
    for card in payload["cards"]:
        if card.get("pose") not in POSE_TEMPLATES:
            card["pose"] = guess_pose(card["object"])

    for card in payload["cards"]:
        cat, old = int(card["category"]), card.get("prompt", "")
        trigger = old.split(" ", 1)[0] if old else TRIGGER

        surface = card.get("surface", "")
        environment = card.get("environment", "")
        if cat == 3 and not surface:
            m = SURFACE_RE.search(old)
            surface = m.group(1) if m else ""
            if not surface:
                notes.append(f"{card['card_id']}: could not recover the surface")
        if cat == 4 and not environment:
            m = ENVIRONMENT_RE.search(old)
            environment = m.group(1) if m else ""
            if not environment:
                notes.append(f"{card['card_id']}: could not recover the environment")

        card["surface"], card["environment"] = surface, environment

        # Sets written before backgrounds were varied get one assigned deterministically —
        # reproducible on every rebuild, but rotated by set and variant so that nine variants
        # do not all come out with the same two backgrounds.
        bg_style = card.get("bg_style", "")
        if cat == 2 and bg_style not in CAT1_BACKGROUNDS:
            # Deterministic per card so a rebuild is reproducible, rotated by set and variant
            # so nine variants do not all come out with the same two walls.
            offset = sum(ord(ch) for ch in str(payload.get("set_id", ""))) + int(payload.get("variant", 0))
            bg_style = CAT2_BACKDROP_ORDER[(offset + int(card["n"])) % len(CAT2_BACKDROP_ORDER)]
        if cat == 1 and not bg_style:
            offset = sum(ord(ch) for ch in str(payload.get("set_id", ""))) + int(payload.get("variant", 0))
            bg_style = CAT1_BACKGROUND_ORDER[(offset + int(card["n"]) - 1) % len(CAT1_BACKGROUND_ORDER)]
        card["bg_style"] = bg_style

        prompt = CATEGORY_TEMPLATES[cat].format(
            trigger=trigger,
            object=POSE_PREFIX[card["pose"]] + str(card["object"]).strip().rstrip("."),
            pose=pose_clause(card["pose"], cat),
            surface=surface or "wooden surface",
            environment=environment or "a themed environment",
            background=cat1_background(pal, bg_style) if cat == 1 else "",
            backdrop=cat2_backdrop(pal, bg_style) if cat == 2 else "",
            bg_primary=pal["bg_primary"], bg_secondary=pal["bg_secondary"],
        )
        # A prompt the producer wrote by hand is not regenerated. Everything else on the card
        # still recompiles, so editing the object of one card cannot silently revert the
        # wording of another.
        if not card.get("prompt_locked"):
            card["prompt"] = " ".join(prompt.split())
        card["negative"] = ", ".join(
            x for x in (NEGATIVE, CATEGORY_NEGATIVE.get(cat, "")) if x)

    cards_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_prompts_csv(cards_json.parent / "prompts.csv", payload["cards"])
    return len(payload["cards"]), notes, calls


EDITABLE = ("name", "object", "pose", "surface", "environment", "bg_style")

# Fields the producer can change that are not part of any prompt. They are saved, and they
# do not trigger a recompile — a note about which film a card nods to cannot alter a picture,
# and treating it as content would silently unlock a hand-written prompt.
METADATA = ("homage",)


def apply_edits(root: Path, edits: list[dict]) -> tuple[int, list[str]]:
    """Apply the producer's changes to a designed set, then recompile its prompts.

    This is the review step: between inventing a set and spending four hours of GPU on it,
    the producer sees every card and can change the object, its pose or the prompt itself.
    Editing content and recompiling — rather than accepting free text everywhere — keeps the
    category, background and style blocks intact, so a corrected object still gets the same
    treatment as the nine cards around it.

    A card whose `prompt` was edited directly is marked and never recompiled again.
    """
    by_id = {e["card_id"]: e for e in edits if e.get("card_id")}
    touched, notes = 0, []
    for cards_json in sorted(Path(root).glob("**/variant_*/cards.json")):
        payload = json.loads(cards_json.read_text(encoding="utf-8"))
        changed = noted = False
        for card in payload["cards"]:
            edit = by_id.get(card["card_id"])
            if not edit:
                continue
            for key in METADATA:
                if key in edit and str(edit[key]).strip() != str(card.get(key, "")).strip():
                    card[key] = str(edit[key]).strip()
                    noted = True
            for key in EDITABLE:
                if key in edit and str(edit[key]).strip() != str(card.get(key, "")).strip():
                    card[key] = str(edit[key]).strip()
                    changed = True
            if card.get("pose") not in POSE_TEMPLATES:
                notes.append(f"{card['card_id']}: unknown pose, falling back to the keyword list")
                card["pose"] = guess_pose(card["object"])
            # A warning, not a veto: at this step the producer is the authority, and a name
            # they typed on purpose is their call. But an edited object goes into the prompt
            # unchanged, so a franchise name arriving this way has to be said out loud.
            for problem in naming_problems(int(card.get("n", 0)), card):
                notes.append(f"{card['card_id']}: {problem.split(': ', 1)[-1]}")
            prompt = str(edit.get("prompt", "")).strip()
            if prompt and prompt != card.get("prompt", "").strip():
                card["prompt"] = prompt
                card["prompt_locked"] = True
                changed = True
            elif changed:
                card.pop("prompt_locked", None)     # content edited: let it recompile
            touched += 1 if changed else 0
        if changed or noted:
            cards_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        if changed:
            _, more, _ = rebuild_variant(cards_json)
            notes += more
    return touched, notes


def rebuild_tree(root: Path, llm: LLM | None = None, repose: bool = False) -> None:
    found = sorted(root.glob("**/variant_*/cards.json"))
    if not found:
        sys.exit(f"No variant_*/cards.json found under {root}")
    total, all_notes, calls = 0, [], 0
    for p in found:
        n, notes, used = rebuild_variant(p, llm, repose)
        total += n
        all_notes += notes
        calls += used
        poses = collections.Counter(
            c["pose"] for c in json.loads(p.read_text(encoding="utf-8"))["cards"])
        print(f"  {p.parent.parent.name}/{p.parent.name}: {n} cards, "
              + ", ".join(f"{k} {v}" for k, v in sorted(poses.items())))
    print(f"\nPrompts rebuilt: {total}, model calls: {calls}")
    for note in all_notes:
        print("  WARNING:", note)


# --------------------------------------------------------------------------- #
# Demo set for --offline (card names stay in Russian: they are product content)
# --------------------------------------------------------------------------- #
OFFLINE = {
    "set_title": "Нуар",
    "concept": "Кабинет частного детектива под дождём: латунь, дым и мокрый асфальт.",
    "palette": {"bg_primary": "deep teal", "bg_secondary": "smoky charcoal",
                "pattern": "soft venetian blind light stripes"},
    "cards": [
        {"name": "Шляпа сыщика", "object": "a grey felt fedora hat", "pose": "three_quarter", "surface": "", "environment": ""},
        {"name": "Лупа", "object": "a brass magnifying glass", "pose": "three_quarter", "surface": "", "environment": ""},
        {"name": "Печатная машинка", "object": "a black vintage typewriter", "pose": "three_quarter", "surface": "", "environment": ""},
        {"name": "Телефон", "object": "a black rotary desk telephone", "pose": "three_quarter", "surface": "", "environment": ""},
        {"name": "Настольная лампа", "object": "a green banker's desk lamp", "pose": "upright",
         "surface": "worn oak desk", "environment": ""},
        {"name": "Граммофон", "object": "a brass horn gramophone", "pose": "three_quarter",
         "surface": "dusty wooden sideboard", "environment": ""},
        {"name": "Досье", "object": "a stack of paper case files tied with string", "pose": "three_quarter",
         "surface": "scratched metal filing cabinet", "environment": ""},
        {"name": "Зонт", "object": "a black open umbrella", "pose": "upright",
         "surface": "", "environment": "a rain-soaked city street at night with glowing street lamps"},
        {"name": "Автомобиль", "object": "a black vintage sedan car", "pose": "upright",
         "surface": "", "environment": "a foggy alley behind a jazz club"},
        {"name": "Микрофон", "object": "a chrome ribbon microphone", "pose": "upright",
         "surface": "", "environment": "a smoky jazz club stage with warm spotlights"},
    ],
}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def demo_cards(base: list[dict], plan: list[int]) -> list[dict]:
    """Fit the built-in demo set to any composition, for --offline.

    The demo is ten cards laid out for the default plan, so its surfaces and environments sit
    at the positions the default plan puts categories 3 and 4 in. Any other composition moves
    those positions, and the cards have to be refitted or validation rejects the set — which
    would make --offline fail for reasons that have nothing to do with the pipeline.
    """
    out = []
    for i, cat in enumerate(plan):
        card = dict(base[i % len(base)])
        if i >= len(base):                      # cycled: keep objects unique
            card["object"] = f"{card['object']} number {i // len(base) + 1}"
            card["name"] = f"{card['name']} {i // len(base) + 1}"
        card["surface"] = "a worn oak desk" if cat == 3 else ""
        card["environment"] = "a dim office at night" if cat == 4 else ""
        out.append(card)
    return out


AMEND_SYSTEM = """You rewrite one image-generation prompt so that a specific defect cannot
happen again. You are told what went wrong with the picture the prompt produced.

Two rules, both learned the hard way on this pipeline:

* Never name the defect. Diffusion models read nouns and drop negations — "no cast shadow"
  produces a cast shadow, "not tipped over" produces a tipped object. State what the picture
  must contain instead.
* Keep everything else. The prompt carries the set's category, background, style and pose
  blocks, and the card has to stay in the same set as its nine neighbours. Change the clause
  that governs the defect; leave the rest word for word.

Reply with the full rewritten prompt as plain text. No commentary, no quotes, no JSON."""

AMEND_USER = """Prompt:
{prompt}

The generated picture was rejected. What is wrong: {defect}
The reviewer's suggested fix: {fix}

Rewrite the prompt."""


def amend_prompt(llm: LLM, prompt: str, defect: str, fix: str) -> str:
    """A prompt rewritten against one defect, or the original if the model is no help.

    Used only after re-seeding has failed twice: two bad seeds in a row means the defect is
    in the wording rather than in the noise, and only then is it worth changing words.
    """
    try:
        out = llm.ask(AMEND_SYSTEM, AMEND_USER.format(prompt=prompt, defect=defect or "unclear",
                                                      fix=fix or "not given")).strip()
    except Exception:
        return prompt
    out = " ".join(out.split())
    # A reply that lost half the prompt is a reply that lost the set's style and category
    # blocks with it. Length is a crude guard, and a crude guard beats a silently broken card.
    if len(out) < len(prompt) * 0.6 or len(out) > len(prompt) * 1.8:
        return prompt
    return out


def parse_slots(text: str) -> dict | None:
    """"2,2,3,3" -> {1: 2, 2: 2, 3: 3, 4: 3}. Empty means the default composition."""
    if not text.strip():
        return None
    parts = [p for p in re.split(r"[,\s]+", text.strip()) if p]
    if len(parts) != 4 or not all(p.isdigit() for p in parts):
        sys.exit("--slots expects four numbers separated by commas, e.g. 2,2,3,3")
    return {i + 1: int(p) for i, p in enumerate(parts)}


def slugify(s: str) -> str:
    """ASCII slug for directory names; transliterates Cyrillic."""
    table = str.maketrans("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                          "abvgdeejzijklmnoprstufhccss_y_eua")
    s = s.lower().translate(table)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:40] or "set"


# --------------------------------------------------------------------------- #
# The stage as a function
# --------------------------------------------------------------------------- #
def design_set(theme: str, wishes: str = "", variants: int = 3, out: str | Path = "sets",
               set_id: str = "", trigger: str = TRIGGER, seed: int = 7, retries: int = 2,
               offline: bool = False, provider: str = "", model: str = "",
               slots: dict | None = None, api_key: str = "", homages: bool = False,
               progress=None) -> dict:
    """Theme and wishes in, a designed set of N variants on disk out. Returns the plan.

    Extracted from main() when the orchestrator appeared: a stage that can only be reached
    through argparse is a script, not a pipeline stage. `progress` takes one line of status
    at a time, which is how the web interface follows a run without parsing stdout.
    """
    say = progress or print
    plan = build_slot_plan(slots)
    if not plan:
        raise RuntimeError("The set composition is empty: at least one card is needed.")
    set_id = set_id or slugify(theme)
    root = Path(out) / set_id
    llm = None if offline else LLM(provider, model, key=api_key)

    used: list[str] = []
    doc = {"theme": theme, "wishes": wishes, "set_id": set_id, "trigger": trigger,
           "slots": {str(c): plan.count(c) for c in (1, 2, 3, 4)},
           "homages": bool(homages), "variants": []}

    for v in range(1, variants + 1):
        say(f"Variant {v} of {variants}: designing the set")
        # Two exit conditions, not one. A set with no problems at all is taken immediately;
        # a set whose only faults are soft is kept as a fallback and the model is asked once
        # more for a tidier one. Only a hard fault throws the reply away — and only a hard
        # fault on every attempt loses the variant.
        data, problems = None, ["not requested"]
        best, best_soft, soft_tries = None, [], 0
        for attempt in range(retries + 1):
            if offline:
                data = json.loads(json.dumps(OFFLINE))
                data["cards"] = demo_cards(data["cards"], plan)
                data["set_title"] = f"{OFFLINE['set_title']} {v}" if v > 1 else OFFLINE["set_title"]
            else:
                try:
                    data = ask_variant(llm, theme, wishes, v, variants, used,
                                       problems if attempt else None, plan, homages)
                except Exception as e:
                    say(f"  attempt {attempt+1}: the request failed — {e}")
                    continue
            hard, soft = validate_split(data, plan, homages)
            problems = hard + soft
            if hard:
                for p in hard:
                    say(f"  attempt {attempt+1}: set rejected — {p}")
                continue
            if best is None or len(soft) < len(best_soft):
                best, best_soft = data, soft
            if not soft:
                break
            for p in soft:
                say(f"  attempt {attempt+1}: {p}")
            # One re-ask on a soft fault, not three. A model that has been told twice that
            # its set is monotonous and has not fixed it will not fix it on the third call
            # either, and the calls are the producer's money.
            soft_tries += 1
            if soft_tries > 1:
                break
        if best is None:
            say(f"  variant {v} skipped: no valid set could be obtained")
            continue
        data = best
        for p in best_soft:
            say(f"  kept with a warning: {p}")

        cards = build_cards(data, set_id, v, trigger, seed + v, plan)
        used += [c["object"].lower() for c in cards]

        vdir = root / f"variant_{v}"
        vdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "set_id": set_id, "variant": v,
            "set_title": data.get("set_title", ""), "concept": data.get("concept", ""),
            "palette": data["palette"], "theme": theme, "wishes": wishes,
            "cards": cards,
        }
        (vdir / "cards.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        write_prompts_csv(vdir / "prompts.csv", cards)

        by_cat = {c: sum(1 for x in cards if x["category"] == c) for c in (1, 2, 3, 4)}
        say(f"  «{data.get('set_title','')}» — {data.get('concept','')}")
        say(f"  categories {by_cat}")
        doc["variants"].append({"variant": v, "set_title": data.get("set_title", ""),
                                "concept": data.get("concept", ""), "dir": str(vdir)})

    if not doc["variants"]:
        raise RuntimeError("Not a single valid variant could be produced.")
    root.mkdir(parents=True, exist_ok=True)
    (root / "plan.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    doc["root"] = str(root)
    return doc


def write_prompts_csv(path: Path, cards: list[dict]) -> None:
    """The columns the generation stage reads, plus `homage` for the producer's records.
    A card carries more fields than that — pose, surface, environment, bg_style — and csv
    refuses unknown keys."""
    fields = ["card_id", "n", "name", "object", "homage", "category", "rarity",
              "prompt", "negative"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: c.get(k, "") for k in fields} for c in cards)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theme", default="", help="set theme")
    ap.add_argument("--rebuild", default="",
                    help="rebuild the prompts of existing sets from their stored content, "
                         "without calling the model. Use it after editing a template: the "
                         "objects stay the same, only their presentation changes")
    ap.add_argument("--wishes", default="", help="free-form wishes from the producer")
    ap.add_argument("--variants", type=int, default=3, help="how many variants of the set (3-5)")
    ap.add_argument("--out", default="sets", help="output root directory")
    ap.add_argument("--set-id", default="", help="set identifier (derived from the theme by default)")
    ap.add_argument("--trigger", default=TRIGGER)
    ap.add_argument("--provider", default="", choices=["", "anthropic", "openai"])
    ap.add_argument("--model", default="")
    ap.add_argument("--seed", type=int, default=7, help="seed for the rarity layout")
    ap.add_argument("--retries", type=int, default=2, help="retries on a malformed model reply")
    ap.add_argument("--repose", action="store_true",
                    help="with --rebuild: ask the model for a pose for every object "
                         "(one request per variant). Content is untouched")
    ap.add_argument("--slots", default="",
                    help="set composition by category, e.g. 2,2,3,3 — the default is the same")
    ap.add_argument("--homages", action="store_true",
                    help=f"allow up to {HOMAGE_LIMIT} cards to be stylised nods to famous "
                         f"films. The reference is recorded in the card's `homage` field and "
                         f"never enters the image prompt")
    ap.add_argument("--offline", action="store_true", help="no API: use the built-in demo set")
    cfg = ap.parse_args()

    if cfg.rebuild:
        rebuild_tree(Path(cfg.rebuild),
                     LLM(cfg.provider, cfg.model) if cfg.repose else None, cfg.repose)
        return
    if not cfg.theme:
        sys.exit("--theme is required (or use --rebuild)")
    if not 1 <= cfg.variants <= 5:
        sys.exit("--variants must be between 1 and 5 (the brief asks for 3-5)")

    try:
        plan = design_set(cfg.theme, cfg.wishes, cfg.variants, cfg.out, cfg.set_id,
                          cfg.trigger, cfg.seed, cfg.retries, cfg.offline,
                          cfg.provider, cfg.model, parse_slots(cfg.slots),
                          homages=cfg.homages)
    except RuntimeError as e:
        sys.exit(f"\n{e}")
    print(f"\nDone: {len(plan['variants'])} variant(s) in {plan['root']}")
    print("Next stage: feed prompts.csv to the ComfyUI batch runner.")


if __name__ == "__main__":
    main()
