#!/usr/bin/env python3
"""
studio.py — the producer's side of the tool: a local page with a form and a gallery.

The pipeline is a library of stages; this is the only file a producer has to know about.
Start it, a browser opens, type a theme and any wishes, press the button. The page shows
which stage is running, how far along it is, and the finished cards as a gallery.

No web framework. `http.server` and a single inline page: adding Flask or Gradio to a tool
that runs on one machine, for one person, next to a 12 GB model, buys nothing and costs an
install step that can fail on someone else's computer.

One run at a time, on purpose — there is one GPU, and a queue would only hide that.

    python studio.py                 # opens http://127.0.0.1:8720
    python studio.py --port 9000 --no-browser
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import orchestrator
import set_designer

STATE = {"run": None, "thread": None, "root": Path("runs").resolve(),
         "gate": None, "set_dir": None}

PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Card set generator</title>
<style>
:root { color-scheme: dark; --bg:#131316; --panel:#1c1c21; --line:#2e2e36;
        --text:#e8e8ee; --dim:#9a9aa6; --accent:#7db3ff; --ok:#6fcf8f; --bad:#ff8686; }
* { box-sizing:border-box }
body { margin:0; background:var(--bg); color:var(--text);
       font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif }
.wrap { max-width:1080px; margin:0 auto; padding:32px 20px 80px }
h1 { font-size:23px; margin:0 0 4px }
.sub { color:var(--dim); margin:0 0 28px }
.card { background:var(--panel); border:1px solid var(--line); border-radius:12px;
        padding:20px; margin-bottom:20px }
label { display:block; font-size:13px; color:var(--dim); margin:14px 0 6px }
label:first-child { margin-top:0 }
input, textarea, select { width:100%; padding:10px 12px; border-radius:8px;
  border:1px solid var(--line); background:#141418; color:var(--text); font:inherit }
textarea { min-height:76px; resize:vertical }
.row { display:flex; gap:16px } .row > div { flex:1 }
button { padding:11px 20px; border-radius:8px; border:0; font:inherit; font-weight:600;
         cursor:pointer; background:var(--accent); color:#0b0b0d }
button.ghost { background:transparent; color:var(--text); border:1px solid var(--line) }
button:disabled { opacity:.45; cursor:default }
.actions { display:flex; gap:10px; align-items:center; margin-top:22px }
.hint { color:var(--dim); font-size:13px }
.stages { display:flex; gap:6px; margin:0 0 14px; flex-wrap:wrap }
.stage { font-size:12px; padding:4px 10px; border-radius:99px; border:1px solid var(--line);
         color:var(--dim) }
.stage.now { border-color:var(--accent); color:var(--accent) }
.stage.done { border-color:#2f4d38; color:var(--ok) }
.bar { height:6px; background:#26262d; border-radius:99px; overflow:hidden }
.bar > i { display:block; height:100%; background:var(--accent); width:0; transition:width .3s }
.log { margin-top:14px; max-height:220px; overflow:auto; font:12.5px/1.6 ui-monospace,
       SFMono-Regular,Consolas,monospace; color:var(--dim); white-space:pre-wrap }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:12px }
.tile { background:#141418; border:1px solid var(--line); border-radius:10px; overflow:hidden }
.tile img { width:100%; display:block; aspect-ratio:832/928; object-fit:cover }
.tile .cap { padding:7px 9px; font-size:12px }
.tile .meta { color:var(--dim); font-size:11px }
.vtitle { margin:26px 0 12px; font-size:16px }
.metrics { display:flex; gap:26px; flex-wrap:wrap; margin-top:6px }
.metric b { display:block; font-size:20px } .metric span { color:var(--dim); font-size:12px }
.err { color:var(--bad); margin-top:10px }
.crow { display:grid; grid-template-columns:26px 1.1fr 1.6fr 130px 34px; gap:8px;
        align-items:center; padding:7px 0; border-bottom:1px solid var(--line) }
.crow input, .crow select { padding:7px 9px; font-size:13px }
.crow .cat { color:var(--dim); font-size:12px; text-align:center }
.crow .more { background:none; border:1px solid var(--line); color:var(--dim);
              padding:6px 8px; border-radius:7px; font-size:12px }
.prompt { grid-column:1 / -1; margin:2px 0 8px }
.prompt textarea { min-height:96px; font:12.5px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace }
.vhead { margin:20px 0 6px; font-size:15px }
.check { display:flex; align-items:center; gap:8px; margin-top:18px; color:var(--text);
         font-size:14px; cursor:pointer }
.check input { width:auto; margin:0 }
.slots { display:grid; grid-template-columns:repeat(4,1fr); gap:10px }
.slots span { display:block; font-size:11.5px; color:var(--dim); margin-bottom:4px }
</style></head><body><div class="wrap">

<h1>Card set generator</h1>
<p class="sub">A theme and a few wishes in, finished card sets out.</p>

<div class="card" id="form">
  <label>Set theme</label>
  <input id="theme" placeholder="Movie collection: film noir" autofocus>
  <label>Free-form wishes <span class="hint">— optional</span></label>
  <textarea id="wishes" placeholder="More brass and rain, a private detective’s props"></textarea>
  <div class="row">
    <div><label>Variants of the set</label>
      <select id="variants">
        <option>1</option><option>2</option><option selected>3</option>
        <option>4</option><option>5</option></select></div>
    <div><label>Candidates per card</label>
      <select id="seeds">
        <option>1</option><option>2</option><option selected>3</option>
        <option>4</option><option>5</option></select></div>
  </div>
  <label>API key <span class="hint">— Anthropic or OpenAI, never written to disk</span></label>
  <input id="key" type="password" placeholder="sk-ant-…" autocomplete="off">
  <label>Set composition <span class="hint" id="slothint"></span></label>
  <p class="hint" style="margin:0 0 8px">Zero is allowed — a set can leave a category out.</p>
  <div class="slots">
    <div><span>1 · floating</span><input id="s1" type="number" min="0" max="10" value="2"></div>
    <div><span>2 · flat backdrop</span><input id="s2" type="number" min="0" max="10" value="2"></div>
    <div><span>3 · on a surface</span><input id="s3" type="number" min="0" max="10" value="3"></div>
    <div><span>4 · in a setting</span><input id="s4" type="number" min="0" max="10" value="3"></div>
  </div>
  <label class="check"><input type="checkbox" id="homages">
    Allow franchise homages <span class="hint">— up to 4 cards may nod to a famous film,
    stylised as our own object. The reference is recorded next to the card and never sent
    to the image generator.</span></label>
  <label class="check"><input type="checkbox" id="judge" checked>
    Check with the judge <span class="hint" id="judgehint"></span></label>
  <div class="actions">
    <button id="go">Build the sets</button>
    <button id="stop" class="ghost" style="display:none">Stop</button>
    <span class="hint" id="estimate"></span>
  </div>
  <div class="err" id="err"></div>
</div>

<div class="card" id="resume" style="display:none">
  <b>Interrupted run</b>
  <p class="hint" style="margin:4px 0 12px">The design is reused and images already rendered
     are skipped, so only what is missing is generated.</p>
  <div id="resumebody"></div>
</div>

<div class="card" id="progress" style="display:none">
  <div class="stages" id="stages"></div>
  <div class="bar"><i id="fill"></i></div>
  <div class="log" id="log"></div>
</div>

<div class="card" id="review" style="display:none">
  <b>Review the set</b>
  <p class="hint" style="margin:4px 0 0">Edit the object, its pose or the prompt itself.
     Changing content recompiles the prompt from the category templates; editing the prompt
     text keeps it exactly as written.</p>
  <div id="reviewbody"></div>
  <div class="actions">
    <button id="gen">Start generation</button>
    <button id="cancel" class="ghost">Cancel run</button>
    <span class="hint" id="revhint"></span>
  </div>
</div>

<div id="results"></div>

<script>
const $ = id => document.getElementById(id);
const NAMES = {design:"design", review:"review", generate:"generate",
               reframe:"frame & background", judge:"judge", retry:"retry",
               assemble:"assemble"};

const SLOTS = ["s1","s2","s3","s4"];
const cardsPerSet = () => SLOTS.reduce((a, id) => a + (+$(id).value || 0), 0);

function estimate() {
  const seeds = +$("seeds").value;
  const per = cardsPerSet();
  const n = +$("variants").value * per * seeds;
  const mins = Math.round(n * 1.6);
  $("estimate").textContent = `${n} images, roughly ` +
    (mins < 90 ? `${mins} min` : `${(mins / 60).toFixed(1)} h`);
  // With one candidate the judge cannot choose, only accept or reject — and a rejection then
  // leaves a hole in the set. Worth saying at the moment the choice is made, not afterwards.
  const missing = SLOTS.filter(id => !+$(id).value).length;
  const off = ["1","2","3","4"].filter((_, i) => !+$(SLOTS[i]).value);
  $("slothint").textContent = per === 0 ? "— empty set"
    : off.length ? `— ${per} cards, without ${off.length > 1 ? "categories" : "category"} ` +
        `${off.join(", ")} · the brief's own sets use all four`
    : `— ${per} cards`;
  $("judgehint").textContent = !$("judge").checked ? "— nothing will be checked"
    : seeds === 1 ? "— nothing to choose between; rejects go to “needs work”"
    : `— picks the best of ${seeds}`;
}
["variants","seeds","judge",...SLOTS].forEach(id =>
  $(id).addEventListener("input", estimate));
estimate();

$("go").onclick = () => start("");

$("stop").onclick = () => fetch("/api/stop", {method:"POST"});
$("cancel").onclick = () => { $("review").style.display = "none";
                              fetch("/api/stop", {method:"POST"}); };

let reviewShown = false;

async function showReview() {
  const data = await (await fetch("/api/cards")).json();
  const poses = data.poses || [];
  let html = "";
  data.variants.forEach(v => {
    html += `<div class="vhead">Variant ${v.variant}${v.set_title ? " — " + v.set_title : ""}
      <div class="hint">${v.concept || ""}</div></div>`;
    v.cards.forEach(c => {
      html += `<div class="crow" data-id="${c.card_id}">
        <div class="cat">${c.category}</div>
        <input class="f-name" value="${esc(c.name)}" placeholder="card name">
        <input class="f-object" value="${esc(c.object)}" placeholder="object in English">
        <select class="f-pose">${poses.map(p =>
          `<option ${p === c.pose ? "selected" : ""}>${p}</option>`).join("")}</select>
        <button class="more" title="prompt">≡</button>
      </div>
      <div class="prompt" style="display:none">
        <div class="hint" style="margin-bottom:4px">Homage — the film this card nods to.
          A note for the set list; it is never sent to the image generator.</div>
        <input class="f-homage" value="${esc(c.homage)}" placeholder="none">
        <textarea class="f-prompt">${esc(c.prompt)}</textarea>
      </div>`;
    });
  });
  $("reviewbody").innerHTML = html;
  $("reviewbody").querySelectorAll(".more").forEach(b => b.onclick = () => {
    const box = b.closest(".crow").nextElementSibling;
    box.style.display = box.style.display === "none" ? "" : "none";
  });
  $("review").style.display = "";
  $("revhint").textContent = "";
}

const esc = s => String(s == null ? "" : s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");

function collectEdits() {
  return [...$("reviewbody").querySelectorAll(".crow")].map(row => ({
    card_id: row.dataset.id,
    name: row.querySelector(".f-name").value,
    object: row.querySelector(".f-object").value,
    pose: row.querySelector(".f-pose").value,
    homage: row.nextElementSibling.querySelector(".f-homage").value,
    prompt: row.nextElementSibling.querySelector(".f-prompt").value,
  }));
}

$("gen").onclick = async () => {
  $("gen").disabled = true;
  $("revhint").textContent = "saving…";
  const r = await (await fetch("/api/cards", {method:"POST",
    body: JSON.stringify(collectEdits())})).json();
  if (r.error) { $("revhint").textContent = r.error; $("gen").disabled = false; return; }
  $("revhint").textContent = r.touched ? `${r.touched} cards updated` : "";
  await fetch("/api/generate", {method:"POST"});
  $("review").style.display = "none";
  $("gen").disabled = false;
};

async function poll() {
  const s = await (await fetch("/api/status")).json();
  if (!s.running) { $("go").disabled = false; $("stop").style.display = "none"; return; }
  const st = s.state;
  $("stages").innerHTML = st.stages.map((name, i) =>
    `<span class="stage ${i < st.stage_index ? "done" : i === st.stage_index ? "now" : ""}">
       ${NAMES[name] || name}</span>`).join("");
  $("fill").style.width = st.total ? (100 * st.done / st.total).toFixed(1) + "%" : "0";
  $("log").textContent = st.log.join("\\n");
  $("log").scrollTop = $("log").scrollHeight;
  if (st.error) $("err").textContent = st.error;
  if (st.stage === "review" && !reviewShown) { reviewShown = true; showReview(); }
  if (st.stage !== "review") reviewShown = false;
  if (st.finished || st.error) {
    $("go").disabled = false; $("stop").style.display = "none";
    if (st.result && st.result.cards) render(st.result);
    return;
  }
  setTimeout(poll, 1500);
}

function render(m) {
  const byVariant = {};
  m.cards.forEach(c => (byVariant[c.variant] = byVariant[c.variant] || []).push(c));
  const mm = m.metrics || {};
  let html = `<div class="card"><b>${m.theme}</b>
    <div class="metrics">
      <div class="metric"><b>${mm.cards_delivered ?? "—"}</b><span>cards</span></div>
      <div class="metric"><b>${mm.first_try_pass_rate != null
          ? Math.round(mm.first_try_pass_rate * 100) + "%" : "—"}</b>
        <span>accepted on the first seed</span></div>
      <div class="metric"><b>${mm.recovered ?? 0}/${mm.retried ?? 0}</b>
        <span>recovered by retry</span></div>
      <div class="metric"><b>${mm.needs_review ?? "—"}</b><span>sent to a human</span></div>
      <div class="metric"><b>${Math.round((mm.wall_time_sec || 0) / 60)} min</b><span>wall time</span></div>
    </div></div>`;
  Object.keys(byVariant).sort().forEach(v => {
    const plan = (m.plan || []).find(p => String(p.variant) === String(v)) || {};
    html += `<div class="vtitle">Variant ${v}${plan.set_title ? " — " + plan.set_title : ""}
      <div class="hint">${plan.concept || ""}</div></div><div class="grid">`;
    byVariant[v].sort((a, b) => a.n - b.n).forEach(c => {
      html += `<div class="tile">
        <img loading="lazy" src="/api/image?path=${encodeURIComponent(c.file)}">
        <div class="cap">${c.name}<div class="meta">cat ${c.category} · ${c.rarity}</div></div>
      </div>`;
    });
    html += `</div>`;
  });
  // Rejected cards are shown, not hidden: a producer needs to see what the judge threw out and
  // why, otherwise the number "6 to review" is just a number.
  if (m.review && m.review.length) {
    const why = {};
    (m.review_reasons || []).forEach(r => why[r.card] = r.reason || r.fix || "");
    html += `<div class="vtitle">Needs work — ${m.review.length}
      <div class="hint">rejected by the judge; not part of the set</div></div><div class="grid">`;
    m.review.forEach(c => {
      html += `<div class="tile" style="border-color:#5a3a3a">
        <img loading="lazy" src="/api/image?path=${encodeURIComponent(c.file)}">
        <div class="cap">${c.name}<div class="meta">variant ${c.variant} · cat ${c.category}</div></div>
      </div>`;
    });
    html += `</div>`;
  }
  $("results").innerHTML = html;
}

async function loadResumable() {
  const { runs } = await (await fetch("/api/resumable")).json();
  if (!runs.length) { $("resume").style.display = "none"; return; }
  $("resumebody").innerHTML = runs.map(r =>
    `<div class="crow" style="grid-template-columns:1fr 150px">
       <div>${esc(r.theme || r.run)}
         <div class="hint">${r.variants} variants · ${r.rendered} images already rendered</div></div>
       <button class="more" data-run="${esc(r.run)}">Continue</button>
     </div>`).join("");
  $("resumebody").querySelectorAll("button").forEach(b => b.onclick = () => start(b.dataset.run));
  $("resume").style.display = "";
}

async function start(resume) {
  $("err").textContent = "";
  if (!resume) {
    if (!$("theme").value.trim()) { $("err").textContent = "A theme is required."; return; }
    if (!cardsPerSet()) { $("err").textContent = "A set needs at least one card."; return; }
  }
  if (!$("key").value.trim()) {
    $("err").textContent = "An API key is required — it pays for designing the set and for the judge.";
    return; }
  const r = await (await fetch("/api/start", {method:"POST", body: JSON.stringify({
    theme:$("theme").value || "resumed run", wishes:$("wishes").value,
    variants:+$("variants").value, seeds:+$("seeds").value,
    judge:$("judge").checked, homages:$("homages").checked,
    key:$("key").value.trim(), resume: resume || "",
    slots:{1:+$("s1").value, 2:+$("s2").value, 3:+$("s3").value, 4:+$("s4").value} })})).json();
  if (r.error) { $("err").textContent = r.error; return; }
  $("go").disabled = true; $("stop").style.display = "";
  $("resume").style.display = "none";
  $("progress").style.display = ""; poll();
}

loadResumable();

fetch("/api/status").then(r => r.json()).then(s => {
  if (s.running) { $("go").disabled = true; $("stop").style.display = "";
                   $("progress").style.display = ""; poll(); }
});
</script></div></body></html>"""


def resumable() -> list[dict]:
    """Runs that have a designed set on disk but no manifest — i.e. they were interrupted.

    Presence of `manifest.json` is the completion mark: it is written last, after assembly.
    Anything designed but unfinished is worth offering back rather than silently leaving on
    disk for a producer to find weeks later.
    """
    out = []
    for cards_json in sorted(STATE["root"].glob("*/*/variant_*/cards.json")):
        set_root = cards_json.parent.parent
        if (set_root / "manifest.json").exists():
            continue
        run_dir = set_root.parent
        payload = json.loads(cards_json.read_text(encoding="utf-8"))
        entry = {"run": run_dir.name, "theme": payload.get("theme", ""),
                 "variants": len(list(set_root.glob("variant_*/cards.json"))),
                 "rendered": len(list(set_root.glob("variant_*/raw/*.png")))}
        if entry not in out:
            out.append(entry)
    return out[-5:]


def clamp(value, low: int = 1, high: int = 5) -> int:
    """Both counts are 1..5. The page offers exactly that range, but the endpoint is reachable
    without the page, and a request for 500 variants should not reach the GPU."""
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return low


def start_run(params: dict) -> dict:
    if STATE["thread"] and STATE["thread"].is_alive():
        return {"error": "A run is already in progress."}
    variants = clamp(params.get("variants", 3))
    seeds = clamp(params.get("seeds", 3))
    raw = params.get("slots") or {}
    slots = {c: clamp(raw.get(str(c), raw.get(c, 0)), 0, 10) for c in (1, 2, 3, 4)}
    if not sum(slots.values()):
        return {"error": "A set needs at least one card."}
    # The key belongs to whoever launched the tool. It is held for the length of the run and
    # written nowhere — not to the run folder, not to the log, not to a config file. Left
    # empty it falls back to the environment, which is how the CLI keeps working unattended.
    key = str(params.get("key", "")).strip()
    if key and not key.isascii():
        return {"error": "The key contains non-Latin characters — that does not look like a key."}
    resume = str(params.get("resume", "")).strip()
    run = orchestrator.Run(params["theme"], params.get("wishes", ""), variants, seeds)
    STATE["run"] = run

    gate = threading.Event()
    STATE["gate"], STATE["set_dir"] = gate, None

    def on_designed(set_dir):
        STATE["set_dir"] = str(set_dir)
        gate.wait()                      # released by /api/generate or by Stop

    def work():
        try:
            orchestrator.produce(
                params["theme"], params.get("wishes", ""), variants, seeds,
                out=STATE["root"], offline=bool(params.get("offline")),
                use_judge=params.get("judge", True) is not False,
                slots=slots, api_key=key,
                homages=bool(params.get("homages")), on_designed=on_designed,
                resume=resume, run=run)
        except Exception as e:
            # The page is the only place a producer looks, so the failure has to arrive there
            # rather than in a terminal they never opened.
            run.error = str(e)
            run.finished = True
            run.say(f"Error: {e}")

    STATE["thread"] = threading.Thread(target=work, daemon=True)
    STATE["thread"].start()
    return {"ok": True}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # The page lives inside the .py file, so it changes whenever the tool is edited.
        # A cached copy then hides the edit and looks like the fix did not work.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if url.path == "/api/status":
            run = STATE["run"]
            return self._json({"running": run is not None,
                               "state": run.as_dict() if run else {}})
        if url.path == "/api/resumable":
            return self._json({"runs": resumable()})
        if url.path == "/api/cards":
            # The designed set, as the review step needs it: content fields plus the compiled
            # prompt, per variant.
            root = STATE.get("set_dir")
            if not root:
                return self._json({"variants": []})
            out = []
            for cards_json in sorted(Path(root).glob("variant_*/cards.json")):
                payload = json.loads(cards_json.read_text(encoding="utf-8"))
                out.append({
                    "variant": payload["variant"],
                    "set_title": payload.get("set_title", ""),
                    "concept": payload.get("concept", ""),
                    "cards": [{k: c.get(k, "") for k in
                               ("card_id", "n", "name", "object", "homage", "pose", "category",
                                "rarity", "surface", "environment", "bg_style", "prompt")}
                              for c in payload["cards"]],
                })
            return self._json({"variants": out, "poses": list(set_designer.POSE_TEMPLATES)})
        if url.path == "/api/image":
            raw = parse_qs(url.query).get("path", [""])[0]
            path = Path(raw).resolve()
            # Serve only from inside the runs directory: the browser sends this path back to
            # us, and a tool that opens any file it is asked for is a tool with a hole in it.
            if not str(path).startswith(str(STATE["root"])) or not path.is_file():
                return self._json({"error": "not found"}, 404)
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            return self._send(200, path.read_bytes(), ctype)
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if url.path == "/api/start":
            size = int(self.headers.get("Content-Length", 0))
            params = json.loads(self.rfile.read(size) or b"{}")
            if not str(params.get("theme", "")).strip():
                return self._json({"error": "A theme is required."})
            return self._json(start_run(params))
        if url.path == "/api/cards":
            size = int(self.headers.get("Content-Length", 0))
            edits = json.loads(self.rfile.read(size) or b"[]")
            root = STATE.get("set_dir")
            if not root:
                return self._json({"error": "Nothing is waiting for review."})
            touched, notes = set_designer.apply_edits(Path(root), edits)
            if STATE["run"]:
                STATE["run"].say(f"Applied edits to {touched} cards")
                for n in notes:
                    STATE["run"].say(f"  warning: {n}")
            return self._json({"ok": True, "touched": touched, "notes": notes})
        if url.path == "/api/generate":
            if STATE.get("gate"):
                STATE["gate"].set()
            return self._json({"ok": True})
        if url.path == "/api/stop":
            if STATE["run"]:
                STATE["run"].stopping = True
                STATE["run"].say("Stopping after the current image…")
            # A run paused for review is not inside the render loop, so the flag alone would
            # never be read: release the gate so it can notice and exit.
            if STATE.get("gate"):
                STATE["gate"].set()
            return self._json({"ok": True})
        self._json({"error": "not found"}, 404)

    def log_message(self, *args) -> None:      # keep the console for the pipeline's own output
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8720)
    ap.add_argument("--out", default="runs", help="where runs are written")
    ap.add_argument("--no-browser", action="store_true")
    cfg = ap.parse_args()

    STATE["root"] = Path(cfg.out).resolve()
    STATE["root"].mkdir(parents=True, exist_ok=True)
    address = f"http://127.0.0.1:{cfg.port}"
    print(f"Tool running at {address}\nClose this window to stop it.")
    if not cfg.no_browser:
        threading.Timer(0.7, lambda: webbrowser.open(address)).start()
    ThreadingHTTPServer(("127.0.0.1", cfg.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
