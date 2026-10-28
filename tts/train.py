"""Training loop shared by every experiment: fixed seed, resume from last.ckpt, best.ckpt by validation loss,
milestone checkpoints, per-epoch CSV log.   CLI: python -m tts.train --config configs/expA.json ..."""
import argparse, json, os, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from tts.data import TTSDataset, collate
from tts.tts_model import TTSModel


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, device, seed):
    """Mean validation losses. The flow loss samples noise and t, so the RNG is fixed: every epoch is scored on the
    same noise and the scores are comparable across epochs."""
    model.eval()                                                             # disables dropout
    # fork_rng: the fixed seed below does not disturb the training RNG stream
    kw = {"device_type": device.type} if device.type in ("cuda", "mps") else {"devices": []}
    totals, n = {"dur": 0.0, "prior": 0.0, "dec": 0.0}, 0
    with torch.random.fork_rng(**kw):
        torch.manual_seed(seed)
        for batch in loader:
            _, parts = model(*[x.to(device) for x in batch])
            for k in totals:
                totals[k] += parts[k]
            n += 1
    model.train()
    return {k: v / n for k, v in totals.items()}


def train(cfg, verbose_every=10):
    """cfg: experiment config + paths (proc_dir, mel_dir, ckpt_dir). Returns the per-epoch log as a DataFrame."""
    # ---- 1. Setup: seed, device, data, model, optimiser ----
    set_seed(cfg["seed"])
    device = pick_device()
    ckpt_dir = Path(cfg["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    proc = Path(cfg["proc_dir"])
    vocab = json.loads((proc / "vocab.json").read_text())
    mel_stats = json.loads((proc / "mel_stats.json").read_text())

    workers = 2 if device.type == "cuda" else 0                              # background loading only helps on CUDA
    g = torch.Generator().manual_seed(cfg["seed"])                           # same shuffle order in every experiment
    train_dl = DataLoader(TTSDataset(proc / "train_subset.json", cfg["mel_dir"], mel_stats), batch_size=cfg["batch_size"],
                          shuffle=True, collate_fn=collate, num_workers=workers, drop_last=True, generator=g)
    val_dl = DataLoader(TTSDataset(proc / "val_monitor.json", cfg["mel_dir"], mel_stats), batch_size=cfg["batch_size"],
                        shuffle=False, collate_fn=collate, num_workers=workers)

    model = TTSModel(len(vocab), decoder=cfg["decoder"], dec_blocks=cfg["dec_blocks"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    milestones = set(cfg["milestones"])

    # ---- 2. Resume from last.ckpt if this experiment was interrupted ----
    start, best_val, best_epoch, log, elapsed = 1, float("inf"), 0, [], 0.0
    last_path = ckpt_dir / "last.ckpt"
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start, best_val, best_epoch, log, elapsed = ck["epoch"] + 1, ck["best_val"], ck["best_epoch"], ck["log"], ck["elapsed_s"]
        torch.set_rng_state(ck["rng_cpu"].cpu())                                     # map_location moved it off the CPU
        print(f"  resumed {cfg['name']} from epoch {ck['epoch']} (best so far {best_val:.4f} @ epoch {best_epoch})")
    if start > cfg["epochs"]:
        print(f"  {cfg['name']}: already trained for {cfg['epochs']} epochs, nothing to do")
        return pd.DataFrame(log)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {cfg['name']}: decoder={cfg['decoder']}, blocks={cfg['dec_blocks']}, params={n_params / 1e6:.2f}M, "
          f"{len(train_dl)} steps/epoch, device={device}")
    # ---- 3. Epoch loop: train on every batch, then score the validation monitor ----
    for epoch in range(start, cfg["epochs"] + 1):
        t0, running = time.time(), []
        for batch in train_dl:
            total, parts = model(*[x.to(device) for x in batch])
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])   # guards against rare large gradients
            opt.step()
            running.append([total.item(), parts["dur"], parts["prior"], parts["dec"]])

        val = evaluate(model, val_dl, device, cfg["seed"])
        val_total = sum(val.values())                                        # same sum of three parts as the training loss
        is_best = val_total < best_val
        if is_best:
            best_val, best_epoch = val_total, epoch
        sec = time.time() - t0
        elapsed += sec
        tr = np.mean(running, axis=0)
        log.append({"epoch": epoch, "train": tr[0], "train_dur": tr[1], "train_prior": tr[2], "train_dec": tr[3],
                    "val": val_total, **{f"val_{k}": v for k, v in val.items()}, "best": is_best, "sec": round(sec, 1)})

        # ---- 4. Checkpoints (self-contained: config, vocab and mel stats travel with the weights) + CSV log ----
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "best_val": best_val,
              "best_epoch": best_epoch, "log": log, "elapsed_s": elapsed, "cfg": cfg, "vocab": vocab,
              "mel_stats": mel_stats, "n_vocab": len(vocab), "rng_cpu": torch.get_rng_state()}
        torch.save(ck, last_path)
        if is_best:
            torch.save(ck, ckpt_dir / "best.ckpt")
        if epoch in milestones:
            torch.save(ck, ckpt_dir / f"epoch_{epoch:03d}.ckpt")
        pd.DataFrame(log).to_csv(ckpt_dir / "train_log.csv", index=False)

        # ---- 5. Progress line: ★ = new best.ckpt, ◆ = milestone checkpoint written ----
        if epoch == start or epoch % verbose_every == 0 or epoch == cfg["epochs"]:
            print(f"  [{cfg['name']}] epoch {epoch:>3}/{cfg['epochs']}  train {tr[0]:.4f}  val {val_total:.4f} "
                  f"(dur {val['dur']:.3f} prior {val['prior']:.3f} dec {val['dec']:.3f})  {sec:.1f}s/epoch"
                  f"{'  ★ best' if is_best else ''}{'  ◆ milestone saved' if epoch in milestones else ''}")
    print(f"  {cfg['name']} done: best val {best_val:.4f} at epoch {best_epoch}, total {elapsed / 60:.1f} min")
    return pd.DataFrame(log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train one experiment from a config file.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--proc_dir", required=True)
    ap.add_argument("--mel_dir", required=True)
    ap.add_argument("--ckpt_root", default="checkpoints/runs")
    a = ap.parse_args()
    c = json.loads(Path(a.config).read_text())
    train({**c, "proc_dir": a.proc_dir, "mel_dir": a.mel_dir, "ckpt_dir": str(Path(a.ckpt_root) / c["name"])})
