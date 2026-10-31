"""bento3-ptv3 train [--key value ...]: every TrainConfig field is a flag (tuples as comma lists, bools as 0/1)."""
import argparse, dataclasses, json
from .config import TrainConfig


def main(argv=None):
    p = argparse.ArgumentParser(prog="bento3-ptv3"); p.add_argument("cmd", choices=["train", "config"])
    for f in dataclasses.fields(TrainConfig):
        p.add_argument("--" + f.name.replace("_", "-"), dest=f.name, default=None)
    a = vars(p.parse_args(argv)); cmd = a.pop("cmd"); d = TrainConfig().to_dict()
    for k, v in a.items():
        if v is None: continue
        t = d[k]
        d[k] = (tuple(type(t[0])(x) if t else float(x) for x in v.split(",")) if v else ()) if isinstance(t, tuple) else \
               bool(int(v)) if isinstance(t, bool) else type(t)(v)
    cfg = TrainConfig(**d)
    if cmd == "config": print(json.dumps(cfg.to_dict(), indent=1)); return
    from .train import train
    print(json.dumps(train(cfg), indent=1))


if __name__ == "__main__":
    main()
