"""Optional side channels. Every call is best effort: the first failure prints one warning and turns that channel off.
Training never waits on them except for the final push.
  Hub     checkpoints + logs + config to hf.co/<hf_repo>/<hf_prefix>/<run>/ (needs HF_TOKEN in the environment); restore on start.
  Tracker Trackio, logged locally in-process; the project is re-published to a free static Space every few minutes."""
import os, time, shutil, threading


class Hub:
    PATTERNS = ["*.json", "*.tsv", "*.txt", "ckpt.pt", "trackio_hf/*.db"]

    def __init__(s, cfg, run_dir):
        s.repo, s.path, s.dir = cfg.hf_repo, f"{cfg.hf_prefix}/{cfg.run_name}", run_dir
        s.token = os.environ.get("HF_TOKEN", ""); s.on = bool(s.token and s.repo); s.th = None
        if not s.on: print("HF sync: off (no HF_TOKEN in the environment); checkpoints stay local", flush=True)

    def snapshot(s):
        """Consistent single-file copy of the Trackio sqlite DB (backup API; safe while Trackio writes) into trackio_hf/."""
        src = os.path.join(s.dir, "trackio")
        if not os.path.isdir(src): return
        try:
            import sqlite3, glob
            os.makedirs(os.path.join(s.dir, "trackio_hf"), exist_ok=True)
            for db in glob.glob(os.path.join(src, "*.db")):
                dst = os.path.join(s.dir, "trackio_hf", os.path.basename(db)); tmp = dst + ".tmp"
                a, b = sqlite3.connect(db), sqlite3.connect(tmp)
                try: a.backup(b); b.execute("PRAGMA journal_mode=DELETE")
                finally: b.close(); a.close()
                os.replace(tmp, dst)
        except Exception as e: print(f"Trackio snapshot skipped ({type(e).__name__}: {e})", flush=True)

    @staticmethod
    def restore_trackio(run_dir):
        import glob
        for db in glob.glob(os.path.join(run_dir, "trackio_hf", "*.db")):
            dst = os.path.join(run_dir, "trackio", os.path.basename(db))
            if not os.path.exists(dst): os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(db, dst)

    def restore(s):
        """Download the run's files from the Hub if its checkpoint is further along than the local one. Returns the step restored or -1."""
        if not s.repo: return -1
        try:
            import torch
            from huggingface_hub import snapshot_download
            tmp = os.path.join(s.dir, "_hf_restore"); shutil.rmtree(tmp, ignore_errors=True)
            snapshot_download(s.repo, local_dir=tmp, allow_patterns=[f"{s.path}/*", f"{s.path}/trackio_hf/*"], token=s.token or None)
            src = os.path.join(tmp, s.path); ck = os.path.join(src, "ckpt.pt")
            if not os.path.exists(ck): shutil.rmtree(tmp, ignore_errors=True); return -1
            st = lambda p: torch.load(p, map_location="cpu", weights_only=False)["step"] if os.path.exists(p) else -1
            remote, local = st(ck), st(os.path.join(s.dir, "ckpt.pt"))
            if remote > local:
                shutil.copytree(src, s.dir, dirs_exist_ok=True); print(f"HF: restored {s.path} at step {remote} (local: {local})", flush=True)
            shutil.rmtree(tmp, ignore_errors=True); return remote if remote > local else -1
        except Exception as e:
            print(f"HF: nothing restored ({type(e).__name__}: {e})", flush=True); return -1

    def push(s, why, wait=False):
        if not s.on: return
        if s.th is not None and s.th.is_alive():
            if not wait: return   # previous upload still running; the next checkpoint will carry this one's files
            s.th.join()
        s.snapshot()
        def go():
            try:
                from huggingface_hub import HfApi
                HfApi(token=s.token).upload_folder(repo_id=s.repo, folder_path=s.dir, path_in_repo=s.path, allow_patterns=s.PATTERNS,
                                                   commit_message=f"{s.path}: {why}")
                print(f"HF: pushed {s.path} ({why})", flush=True)
            except Exception as e:
                print(f"HF: push failed ({type(e).__name__}: {e}); will retry at the next checkpoint", flush=True)
        s.th = threading.Thread(target=go, daemon=True); s.th.start()
        if wait: s.th.join(900)


class Tracker:
    def __init__(s, cfg, run_dir, config):
        s.on, s.run, s.space, s.pub_t, s.pub_th, s.cfg = False, None, cfg.trackio_space or None, 0.0, None, cfg
        s.dir = os.path.join(run_dir, "trackio"); os.environ["TRACKIO_DIR"] = s.dir
        if s.space and not os.environ.get("HF_TOKEN"): s.space = None; print("Trackio: no HF_TOKEN, logging locally only (no public Space)", flush=True)
        try:
            import trackio
            s.T = trackio; s.run = trackio.init(project=cfg.trackio_project, name=cfg.run_name, config=config, resume="allow", space_id=None,
                                                  embed=False, auto_log_gpu=False, auto_log_cpu=False)
            s.on = True; s.next = int(getattr(s.run, "_next_step", 0) or 0)
            if s.space: print(f"Trackio dashboard: https://huggingface.co/spaces/{s.space} (project {cfg.trackio_project}, run {cfg.run_name})", flush=True)
        except Exception as e: print(f"Trackio: off ({type(e).__name__}: {e})", flush=True)

    def log(s, m, step):
        if not s.on or step < getattr(s, "next", 0) and not any(k.startswith(("eval", "probe", "base", "gene")) for k in m): return
        try: s.run.log({k: float(v) for k, v in m.items() if isinstance(v, (int, float))}, step=int(step)); s.publish()
        except Exception as e: s.on = False; print(f"Trackio: off ({type(e).__name__}: {e}); training continues", flush=True)

    def publish(s, force=False, wait=0):
        if not (s.on and s.space) or (s.pub_th is not None and s.pub_th.is_alive()): return
        if not force and time.time() - s.pub_t < s.cfg.trackio_publish_min * 60: return
        s.pub_t = time.time()
        def go():
            try: s.T.sync(project=s.cfg.trackio_project, space_id=s.space, sdk="static", force=True)
            except FileNotFoundError: s.pub_t = 0.0
            except Exception as e: print(f"Trackio: publishing off ({type(e).__name__}: {e})", flush=True); s.space = None
        s.pub_th = threading.Thread(target=go, daemon=True); s.pub_th.start()
        if wait: s.pub_th.join(wait)

    def finish(s):
        if s.on:
            try: s.run.finish()
            except Exception: pass
        s.publish(force=True, wait=300)
