import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
TRAIN_DIR = PARENT / "archive" / "data" / "data" / "train"
OUT = PARENT / "EXP006_colab_workspace.zip"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return path


amendment = json.loads(require(ROOT / "EXP006_DEVICE_AMENDMENT.json").read_text())
selected_path = require(
    ROOT / "results" / "v2_exp006_gcn_tuned" / "screen" / "selected_top2.json"
)
selected = json.loads(selected_path.read_text())
if selected != amendment["locked_shortlist"]:
    raise RuntimeError("Local selected_top2.json does not match the locked shortlist")

exp002 = require(ROOT / "results" / "v2_exp002_tcn_gcn_fixed" / "all_runs.csv")
train_files = sorted(TRAIN_DIR.glob("*.npy"))
if len(train_files) != 82:
    raise RuntimeError(f"Expected 82 TRAIN arrays including excluded T-10, found {len(train_files)}")

project_files = []
for p in sorted((ROOT / "src").glob("*.py")):
    project_files.append(p)
for p in sorted((ROOT / "configs").glob("*")):
    if p.is_file():
        project_files.append(p)

for name in [
    "EXP006_PRETRAIN_FREEZE.json",
    "EXP006_DEVICE_AMENDMENT.json",
    "EXP006_PROTOCOL.md",
    "EXP006_RUN_INSTRUCTIONS.txt",
    "EXP006_device_amendment.md",
    "exp006_device_patch.diff",
]:
    p = ROOT / name
    if p.exists():
        project_files.append(p)

screen_dir = ROOT / "results" / "v2_exp006_gcn_tuned" / "screen"
for p in sorted(screen_dir.rglob("*")):
    if p.is_file():
        project_files.append(p)
project_files.append(exp002)

if OUT.exists():
    OUT.unlink()

with ZipFile(OUT, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
    for p in project_files:
        rel = p.relative_to(ROOT)
        arc = Path("workspace") / "smap_msl_tcn_gcn_v2" / rel
        zf.write(p, arcname=str(arc))

    for p in train_files:
        arc = Path("workspace") / "archive" / "data" / "data" / "train" / p.name
        zf.write(p, arcname=str(arc))

    manifest = {
        "purpose": "EXP006 Colab TRAIN-only transfer",
        "project_root": "workspace/smap_msl_tcn_gcn_v2",
        "train_arrays": len(train_files),
        "nasa_test_included": False,
        "anomaly_labels_included": False,
        "locked_shortlist": amendment["locked_shortlist"],
        "parent_freeze_id": amendment["parent_freeze_id"],
        "device_amendment_id": amendment["amendment_id"],
    }
    zf.writestr(
        "workspace/EXP006_COLAB_TRANSFER_MANIFEST.json",
        json.dumps(manifest, indent=2) + "\n",
    )

print(f"Created: {OUT}")
print(f"Size: {OUT.stat().st_size / (1024**2):.1f} MiB")
print(f"SHA256: {sha256(OUT)}")
print("Contains TRAIN arrays only; NASA TEST arrays and anomaly labels are excluded.")
