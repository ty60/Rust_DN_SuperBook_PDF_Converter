# AI Bridge Setup on macOS (Apple Silicon)

Reproducible setup for `superbook-pdf/ai_bridge/` dependencies on macOS.
Working directory for all commands: `superbook-pdf/ai_bridge/`.

## Why this is not just `pip install -r requirements.txt`

Three things break the naive install:

1. **Python version** — Homebrew's default `python3` may be 3.14+, which is too new for `basicsr`. We need 3.12.
2. **`basicsr` build isolation** — `basicsr`'s `setup.py` imports `torch` at build time. Pip's default build isolation uses an empty env, so the build fails with `Failed to build 'basicsr' when getting requirements to build wheel`.
3. **`torchvision` API removal** — `basicsr` imports `torchvision.transforms.functional_tensor`, which was removed in torchvision ≥0.17. Current torchvision is 0.26, so runtime import crashes.
4. **CUDA index URL in README** — the `--index-url https://download.pytorch.org/whl/cu121` from the README is Linux/Windows + NVIDIA only. On Mac, omit it to get the default build (CPU + MPS).

## Steps

```bash
cd superbook-pdf/ai_bridge

# 1. Create a clean Python 3.12 venv (install via `brew install python@3.12` if missing)
python3.12 -m venv .venv

# 2. Upgrade build tools
.venv/bin/pip install --upgrade pip setuptools wheel

# 3. Install torch + torchvision FIRST (no CUDA index on Mac — default build gives CPU + MPS)
.venv/bin/pip install torch torchvision numpy opencv-python

# 4. Install basicsr without build isolation (so its setup.py can find torch)
.venv/bin/pip install basicsr --no-build-isolation

# 5. Install the rest of requirements.txt
.venv/bin/pip install -r requirements.txt
```

## Post-install patch

`basicsr/data/degradations.py` references a removed torchvision API. Fix it in the venv:

```bash
# Path: .venv/lib/python3.12/site-packages/basicsr/data/degradations.py, line 8
# Change:
#   from torchvision.transforms.functional_tensor import rgb_to_grayscale
# To:
#   from torchvision.transforms.functional import rgb_to_grayscale
```

One-liner (safer than `sed -i ''` which corrupted the file once in testing — use python):

```bash
.venv/bin/python -c "
import pathlib
p = pathlib.Path('.venv/lib/python3.12/site-packages/basicsr/data/degradations.py')
p.write_text(p.read_text().replace(
    'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
    'from torchvision.transforms.functional import rgb_to_grayscale'
))
"
```

## Verify

```bash
.venv/bin/python -c "import torch, torchvision, basicsr, realesrgan, yomitoku; print('torch', torch.__version__, '| mps:', torch.backends.mps.is_available()); print('basicsr', basicsr.__version__); print('yomitoku', yomitoku.__version__); print('ALL OK')"
```

Expected output (versions may differ):

```
torch 2.11.0 | mps: True
basicsr 1.4.2
yomitoku 0.12.0
ALL OK
```

## Using the venv

Either activate it:

```bash
source superbook-pdf/ai_bridge/.venv/bin/activate
```

Or invoke the interpreter directly:

```bash
superbook-pdf/ai_bridge/.venv/bin/python yomitoku_bridge.py ...
```

## Notes

- The `yomitoku_bridge.py` script auto-detects MPS on Mac (added in commit `21aab8d`), so OCR will use the Apple Silicon GPU instead of CPU.
- If you hit MPS op-not-supported errors, set `PYTORCH_ENABLE_MPS_FALLBACK=1` to silently fall back to CPU for unsupported ops.
- If you only need YomiToku OCR (not RealESRGAN upscaling), you can skip basicsr/realesrgan/facexlib/gfpgan and avoid all the basicsr-related pain.
