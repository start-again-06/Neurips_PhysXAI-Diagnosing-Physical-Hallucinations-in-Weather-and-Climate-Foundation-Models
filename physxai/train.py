"""Train PhysXAI on synthetic clean/perturbed states and report held-out AUROC.

Usage:
    python -m physxai.train --config configs/base.yaml --out physxai.pt
"""
import argparse
import dataclasses
import json

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import load_config
from .data.synth_generator import make_training_batch, sample_clean
from .grid import Grid
from .losses.multi_task_loss import MultiTaskLoss
from .metrics import auroc
from .models.physxai import PhysXAI


def evaluate(model, data_grid, gen, n_batches, batch_size, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for _ in range(n_batches):
            b = make_training_batch(batch_size, data_grid, gen)
            out = model(b["x"].to(device), b["prev"].to(device), b["dt"])
            scores.append(out["h"].cpu())
            labels.append(b["y"])
    model.train()
    return auroc(torch.cat(scores), torch.cat(labels))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="physxai.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=None, help="override epochs * steps_per_epoch")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tc = cfg.train
    torch.manual_seed(tc.seed)
    gen = torch.Generator().manual_seed(tc.seed)
    eval_gen = torch.Generator().manual_seed(tc.seed + 1)

    # Data is generated on the CPU; the model grid is moved to the training device.
    data_grid = Grid(cfg.grid.n_lat, cfg.grid.n_lon, cfg.grid.pressures_hpa)
    grid = Grid(cfg.grid.n_lat, cfg.grid.n_lon, cfg.grid.pressures_hpa).to(args.device)
    model = PhysXAI(cfg.model, grid).to(args.device)
    model.fit_statistics(sample_clean(256, data_grid, gen).to(args.device))

    loss_fn = MultiTaskLoss(tc.tau).to(args.device)
    opt = AdamW(list(model.parameters()) + list(loss_fn.parameters()),
                lr=tc.lr, weight_decay=tc.weight_decay)
    total = args.steps or tc.epochs * tc.steps_per_epoch
    sched = CosineAnnealingLR(opt, T_max=total)

    for step in range(1, total + 1):
        b = make_training_batch(tc.batch_size, data_grid, gen)
        b = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = model(b["x"], b["prev"], b["dt"])
        loss, parts = loss_fn(out, b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % tc.log_every == 0 or step == total:
            record = {"step": step, "loss": round(float(loss.detach()), 4)}
            record.update({k: round(v, 4) for k, v in parts.items()})
            print(json.dumps(record), flush=True)

    auc = evaluate(model, data_grid, eval_gen, tc.eval_batches, tc.batch_size, args.device)
    print(f"held-out AUROC (synthetic): {auc:.3f}", flush=True)
    torch.save({
        "model": model.state_dict(),
        "loss": loss_fn.state_dict(),
        "config": dataclasses.asdict(cfg),
        "auroc": auc,
    }, args.out)
    return auc


if __name__ == "__main__":
    main()
