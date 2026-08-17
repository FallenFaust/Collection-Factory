# Pointing ComfyUI at the FluxGym models

FluxGym has already downloaded everything the validation run needs — the Flux transformer,
both CLIP encoders and the VAE — and it writes trained LoRA checkpoints to its `outputs`
directory. None of it has to be downloaded or copied again (~24 GB saved); ComfyUI reads the
files where they are.

## This machine

* ComfyUI base directory: `E:\projects\Card generation\NotOne\ComfyUI` (ComfyUI 0.33.1,
  Desktop standalone environment, own `.venv`).
* The application binaries live separately in `E:\projects\Card generation\Comfy Desktop`
  and contain no models — that directory is not involved in this setup.
* `extra_model_paths.yaml` has been placed in the base directory. Done; no further edits.

## Why not the AppData config

ComfyUI Desktop keeps a model configuration at
`%APPDATA%\ComfyUI\extra_models_config.yaml`, written by the installer. Editing it works, but
it is a file the installer owns, and a YAML slip there takes down every model path at once.

It is also unnecessary. `main.py` (lines 130-132) loads a plain `extra_model_paths.yaml` from
the directory containing `main.py`, regardless of build:

```python
extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
if os.path.isfile(extra_model_paths_config_path):
    utils.extra_config.load_extra_path_config(extra_model_paths_config_path)
```

So the extra paths live in their own file, next to the installer's config rather than inside
it, and can be deleted to revert.

## The mapping

```yaml
fluxgym:
    base_path: E:/pinokio/api/fluxgym.git
    diffusion_models: models/unet
    unet: models/unet
    clip: models/clip
    vae: models/vae
    loras: outputs
```

`diffusion_models` and `unet` both appear because ComfyUI renamed the category and kept the
old name as an alias; listing both works on either version.

Those FluxGym paths are Windows junctions created by Pinokio's `fs.link`, pointing into its
shared model drive. ComfyUI resolves them like ordinary directories.

## Verifying

Restart the app fully, then:

```bash
python lora_grid_test.py --list
```

On this machine ComfyUI Desktop listens on **8188**, the same default the portable build and
these scripts use, so no `--url` is needed. Do not assume it: the Desktop build is documented
as preferring 8000 and falls back to an arbitrary free port when its first choice is taken.
If the runner reports it cannot connect, find the real port with:

```powershell
Get-Process python*,*omfy* -ErrorAction SilentlyContinue |
  ForEach-Object { Get-NetTCPConnection -OwningProcess $_.Id -State Listen -ErrorAction SilentlyContinue } |
  Select-Object LocalAddress,LocalPort,OwningProcess
```

then pass `--url http://127.0.0.1:<port>`.

Expected:

* **UNET** — `flux1-dev.sft`
* **CLIP** — `t5xxl_fp16.safetensors` and `clip_l.safetensors`
* **VAE** — `ae.sft`
* **LoRA** — `cardcollectionsg1/...safetensors`, one entry per saved epoch

FluxGym writes one subdirectory per training run, so LoRA entries appear as
`cardcollectionsg1/cardcollectionsg1-000004.safetensors`. That is expected; `--lora-filter`
matches substrings and handles it.

`.sft` is Black Forest Labs' extension for safetensors files. ComfyUI accepts it natively —
no renaming needed.

If a category comes back empty, `base_path` is wrong or the app was not restarted. If FluxGym
never finished downloading a model, its folder is simply empty — check the corresponding
directory under `E:\pinokio\api\fluxgym.git\models` in Explorer.

## Other ComfyUI flavours

**Portable / git clone** — same file, same place: the ComfyUI root next to `main.py`.

**Pinokio-installed ComfyUI** — nothing to do. FluxGym's `install.js` declares a shared drive
and lists ComfyUI among its peers, so the model folders and the LoRA output directory are
linked automatically:

```js
drive: { vae: "models/vae", clip: "models/clip", unet: "models/unet", loras: "outputs" }
peers: [ pinokiofactory/comfy.git, cocktailpeanutlabs/comfyui.git, ... ]
```

## Then

```bash
python lora_grid_test.py --lora-filter cardcollectionsg1 --dry-run
python lora_grid_test.py --lora-filter cardcollectionsg1 --weights 0,0.6,0.8,1.0
```

See `README_lora_test.md` for how to read the resulting contact sheets.
