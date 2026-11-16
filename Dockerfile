FROM python:3.15-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

RUN pip install --no-cache-dir packaging
COPY pyproject.toml poetry.lock ./

# Linux wheels on PyPI are the CUDA build. Take torch from the CPU index.
RUN python - <<'PY'
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.markers import Marker, default_environment

lock = tomllib.load(open("poetry.lock", "rb"))
env = default_environment()
lines = []
torch_version = None

def marker_for(pkg):
    marker = pkg.get("markers")
    if isinstance(marker, dict):
        return marker.get("main")
    return marker

def cuda_stack(name):
    return name == "torch" or name == "triton" or name.startswith(("nvidia-", "cuda-"))

for pkg in lock["package"]:
    if "main" not in (pkg.get("groups") or ["main"]):
        continue
    name = pkg["name"]
    if name == "torch":
        torch_version = pkg["version"]
    marker = marker_for(pkg)
    if marker not in (None, "", "<empty>") and not Marker(marker).evaluate(env):
        continue
    if cuda_stack(name):
        continue
    parts = [f"{name}=={pkg['version']}"]
    parts.extend(f"    --hash={item['hash']}" for item in pkg["files"])
    lines.append(" \\\n".join(parts))

if torch_version is None:
    raise SystemExit("poetry.lock has no torch")
cpu_version = torch_version if torch_version.endswith("+cpu") else f"{torch_version}+cpu"
Path("/tmp/requirements.txt").write_text("\n".join(lines) + "\n")
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--no-deps",
        "--require-hashes",
        "-r",
        "/tmp/requirements.txt",
    ]
)
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--no-deps",
        "--index-url",
        "https://download.pytorch.org/whl/cpu",
        f"torch=={cpu_version}",
    ]
)
import torch

if not torch.__version__.startswith(cpu_version):
    raise SystemExit(f"expected {cpu_version}, got {torch.__version__}")
PY

RUN python - <<'PY'
from transformers import AutoModelForSequenceClassification, AutoTokenizer

name = "snunlp/KR-FinBert-SC"
AutoTokenizer.from_pretrained(name)
AutoModelForSequenceClassification.from_pretrained(name)
PY

COPY scoring.py finbert.py app.py ./
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
