"""Build a portable follow-up demo and a clean project ZIP from this checkout.

Generated deliverables go to ../deliverables. Only the listed project folders are
selected, so personal working files, caches and repository metadata never are. Requires nbformat (requirements-dev).
"""
from pathlib import Path
import base64
import copy
import hashlib
import io
import json
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT.parent / "deliverables"
SKIP = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}


def selected_files():
    result = [ROOT / n for n in ("README.md", "requirements.txt", "requirements-dev.txt",
                                  "conftest.py", "pytest.ini", ".gitignore")]
    for folder in ("assets", "src", "tests", "notebooks", "outputs", "tools"):
        result.extend(p for p in (ROOT / folder).rglob("*")
                      if p.is_file() and not SKIP.intersection(p.parts)
                      and p.suffix not in (".pyc", ".pyo"))
    # The reviewer needs the verification tool, not this packaging script.
    return sorted(p for p in result if p != Path(__file__).resolve())


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    files = selected_files()
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            rel = p.relative_to(ROOT)
            if (rel.parts[0] in ("src", "assets", "tests")
                    or rel.name in ("conftest.py", "pytest.ini", "requirements.txt")
                    or (rel.parts[0] == "outputs" and rel.suffix == ".csv" and "followup_" in rel.name)):
                z.write(p, rel.as_posix())
    payload = memory.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    bootstrap = '''import base64, hashlib, importlib.util, io, os, subprocess, sys, tempfile, zipfile
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules
base = Path(os.environ.get("AULECV_DEMO_WORKDIR", "/content" if IN_COLAB else str(Path.cwd())))
base.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT = Path(tempfile.mkdtemp(prefix="aule_followup_", dir=base))
PAYLOAD_B64 = PAYLOAD_TOKEN
payload = base64.b64decode(PAYLOAD_B64)
assert hashlib.sha256(payload).hexdigest() == DIGEST_TOKEN, "Embedded payload is damaged"
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for item in archive.infolist():
        target = (PROJECT_ROOT / item.filename).resolve()
        if not target.is_relative_to(PROJECT_ROOT.resolve()):
            raise ValueError("Invalid embedded path")
    archive.extractall(PROJECT_ROOT)
required = {"cv2": "opencv-python-headless", "numpy": "numpy", "pandas": "pandas",
            "matplotlib": "matplotlib", "imageio": "imageio", "PIL": "pillow"}
missing = [p for m, p in required.items() if importlib.util.find_spec(m) is None]
if missing and IN_COLAB:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
elif missing:
    raise ImportError("Install required packages: " + ", ".join(missing))
os.environ["AULECV_PROJECT_ROOT"] = str(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# A repeated Run all must load this payload rather than an older imported copy.
for name in list(sys.modules):
    if name == "aulecv" or name.startswith("aulecv."):
        del sys.modules[name]
import numpy as np, cv2, pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from aulecv import config as C, detect, rotation, navigation, camera, viz, closed_loop as cl
from aulecv import io as aio
np.random.seed(C.RANDOM_SEED)
plt.rcParams["figure.dpi"] = 110
OUT = aio.ensure_output_dirs()
print("Embedded project loaded; Python", sys.version.split()[0], "OpenCV", cv2.__version__)
'''.replace("PAYLOAD_TOKEN", repr(encoded)).replace("DIGEST_TOKEN", repr(digest))
    demo = nbformat.v4.new_notebook()
    demo.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
    demo.cells = [nbformat.v4.new_markdown_cell(
        "# AULE follow-up: image-based control for Tasks B and D\n\n"
        "Choose **Run all**. The source and reference image are embedded; no Drive mount is needed.\n\n"
        "The examples run live. `FULL_RUNS = False` loads clearly labelled, previously computed many-run "
        "tables; set it to `True` to recompute those studies (several minutes). "
        "The full ZIP also includes independent hold-out checks and regression tests.\n\n"
        "This is a planar-scene simulation, not a real-camera or flight-control validation."),
        nbformat.v4.new_code_cell(bootstrap),
        nbformat.v4.new_code_cell("FULL_RUNS = False  # True reruns the randomised studies below")]
    for filename in ("05_closed_loop_crop_navigation.ipynb", "06_closed_loop_camera_return.ipynb"):
        nb = nbformat.read(ROOT / "notebooks" / filename, as_version=4)
        for i, cell in enumerate(nb.cells):
            if i == 1:  # replace the project-finding setup with the embedded bootstrap
                continue
            cell = copy.deepcopy(cell)
            cell.metadata = {}
            if cell.cell_type == "code":
                cell.outputs, cell.execution_count = [], None
                if filename.startswith("05") and "def compare(view" in cell.source:
                    cell.source = "if FULL_RUNS:\n" + "\n".join("    " + line for line in cell.source.splitlines()) + '''
else:
    print("Previously computed randomised Part B study; set FULL_RUNS=True to recompute")
    summary = pd.read_csv(aio.output_path("crop_navigation", "followup_closed_loop_summary.csv"))
    display(summary)
'''
                if filename.startswith("06") and "for k in range(8):" in cell.source:
                    cell.source = "if FULL_RUNS:\n" + "\n".join("    " + line for line in cell.source.splitlines()) + '''
else:
    print("Previously computed randomised Part D study; set FULL_RUNS=True to recompute")
    runs = pd.read_csv(aio.output_path("camera_return", "followup_closed_loop_runs.csv"))
    display(runs)
'''
            demo.cells.append(cell)
    nbformat.validate(demo)
    demo_path = DEST / "AULE_FollowUp_Verified_Demo.ipynb"
    nbformat.write(demo, demo_path)

    zip_path = DEST / "AULE_ComputerVisionChallenge_verified_2026-10-07.zip"
    manifest = {}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            name = ROOT.name + "/" + p.relative_to(ROOT).as_posix()
            z.write(p, name)
            manifest[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(name)).hexdigest() == sha for name, sha in manifest.items())
        allowed = {"assets", "src", "tests", "notebooks", "outputs", "tools", "README.md",
                   "requirements.txt", "requirements-dev.txt", "conftest.py", "pytest.ini", ".gitignore"}
        assert all(name.split("/")[1] in allowed and "__pycache__/" not in name for name in z.namelist())
    (DEST / "submission_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"ZIP: {zip_path.name}: {len(manifest)} files, every hash verified")
    print(f"Demo: {demo_path.name}; embedded payload SHA256 {digest}")


if __name__ == "__main__":
    main()
