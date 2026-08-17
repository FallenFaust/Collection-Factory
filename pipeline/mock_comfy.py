"""Mock ComfyUI for exercising lora_grid_test.py without a GPU. Not for production use.

Implements just enough of the API surface the runner touches: /system_stats, /object_info,
/prompt, /history/<id> and /view. Generated images are flat colour fields stamped with the
LoRA name, strength and prompt, which is enough to verify wiring and sheet assembly.
"""
import io, json, random
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from PIL import Image, ImageDraw

OBJ = {
    "UNETLoader": {"input": {"required": {
        "unet_name": [["flux1-dev.safetensors", "sd35.safetensors"], {}],
        "weight_dtype": [["default", "fp8_e4m3fn"], {}]}}},
    "DualCLIPLoader": {"input": {"required": {
        "clip_name1": [["t5xxl_fp16.safetensors", "clip_l.safetensors"], {}],
        "clip_name2": [["t5xxl_fp16.safetensors", "clip_l.safetensors"], {}],
        "type": [["flux", "sdxl"], {}]}}},
    "VAELoader": {"input": {"required": {"vae_name": [["ae.safetensors"], {}]}}},
    "LoraLoaderModelOnly": {"input": {"required": {"lora_name": [[
        "cardcollectionsg1-000004.safetensors",
        "cardcollectionsg1-000008.safetensors",
        "cardcollectionsg1-000012.safetensors",
        "cardcollectionsg1.safetensors",
        "other_style.safetensors"], {}]}}},
}
HIST, IMGS = {}, {}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/system_stats":
            return self._send(200, b'{"system":{"os":"mock"}}')
        if u.path == "/object_info":
            return self._send(200, json.dumps(OBJ).encode())
        if u.path.startswith("/history/"):
            pid = u.path.rsplit("/", 1)[-1]
            return self._send(200, json.dumps({pid: HIST[pid]} if pid in HIST else {}).encode())
        if u.path == "/view":
            fn = parse_qs(u.query)["filename"][0]
            return self._send(200, IMGS[fn], "image/png")
        self._send(404, b"{}")

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        g = json.loads(self.rfile.read(n))["prompt"]
        pid = f"p{len(HIST)+1}"
        lora = g.get("4", {}).get("inputs", {})
        txt = g["5"]["inputs"]["text"]
        w, h = g["7"]["inputs"]["width"], g["7"]["inputs"]["height"]
        im = Image.new("RGB", (w, h), (random.randint(30, 90),) * 3)
        d = ImageDraw.Draw(im)
        d.text((20, 20), f"lora={lora.get('lora_name','NONE')}\nstr={lora.get('strength_model',0)}\n{txt[:60]}",
               fill=(255, 255, 0))
        buf = io.BytesIO(); im.save(buf, "PNG")
        fn = f"{pid}.png"
        IMGS[fn] = buf.getvalue()
        HIST[pid] = {"status": {"status_str": "success"},
                     "outputs": {"11": {"images": [{"filename": fn, "subfolder": "", "type": "output"}]}}}
        self._send(200, json.dumps({"prompt_id": pid}).encode())


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8199), H).serve_forever()
