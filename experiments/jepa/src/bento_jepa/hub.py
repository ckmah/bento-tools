"""Getting phase inputs onto a fresh machine (e.g. molab) and results off it. All storage is the private HF repo.

python -m bento_jepa.hub prepare WORK --scale base   data arrays + nucleus/cell polygons (private HF repo, else rebuilt from
                                                     the public 10x bundle), the frozen split (rebuilt, sha256 checked), and
                                                     the CellWorld checkpoint (sha256 checked against its manifest)
python -m bento_jepa.hub push WORK/phase1a --prefix jepa/phase1a/base   upload a results folder to the private HF repo
"""
import os, sys, json, hashlib, argparse, subprocess, types
from importlib import resources

REPO = "ckmah/bento3-ptv3-runs"                    # private; HF_TOKEN must be set
CW_REPO = "Haiping-UoM/CellWorld"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def prepare(work, scale):
    from bento3_ptv3.data import ensure_data, ensure_boundaries
    from huggingface_hub import hf_hub_download
    cfg = types.SimpleNamespace(data_dir=os.path.join(work, "data"), data_source="hf", hf_data_repo=REPO, hf_data_path="data/xenium_breast_idc_v1")
    data = ensure_data(cfg); ensure_boundaries(data); print(f"data: {data}", flush=True)
    p0 = os.path.join(work, "phase0"); want = json.loads(resources.files("bento_jepa").joinpath("assets/split.json").read_text())["role_sha256"]
    if not (os.path.exists(os.path.join(p0, "role.npy")) and sha(os.path.join(p0, "role.npy")) == want):
        subprocess.run([sys.executable, "-m", "bento_jepa.split", data, p0], check=True)
    got = sha(os.path.join(p0, "role.npy")); assert got == want, f"split differs from phase 0: {got} != {want}"; print("split: sha256 OK", flush=True)
    cw = os.path.join(work, "cellworld"); man = json.load(open(hf_hub_download(CW_REPO, "manifest.json", local_dir=cw)))["models"][scale]
    ck = hf_hub_download(CW_REPO, man["checkpoint"], local_dir=cw)
    assert sha(ck) == man["checkpoint_sha256"], "CellWorld checkpoint sha256 mismatch"; print(f"checkpoint: {ck} sha256 OK", flush=True)
    json.dump(dict(data=data, phase0=p0, ckpt=ck), open(os.path.join(work, "paths.json"), "w"))


def push(folder, prefix):
    from huggingface_hub import HfApi
    api = HfApi(); api.create_repo(REPO, private=True, exist_ok=True)
    api.upload_folder(repo_id=REPO, folder_path=folder, path_in_repo=prefix, commit_message=f"{prefix}: results")
    print(f"pushed {folder} -> hf.co/{REPO}/{prefix} (private)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("prepare"); a1.add_argument("work"); a1.add_argument("--scale", default="base")
    a2 = sub.add_parser("push"); a2.add_argument("folder"); a2.add_argument("--prefix", required=True)
    a = ap.parse_args(); prepare(a.work, a.scale) if a.cmd == "prepare" else push(a.folder, a.prefix)
