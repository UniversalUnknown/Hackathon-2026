#!/usr/bin/env python3
"""Train the ZNCC candidate re-ranker (see ml/ranker.py).

Improvements over baseline:
  - Focal Loss (gamma=2, alpha=0.25) to focus on hard negatives
  - Cosine annealing LR with warm-up
  - Gradient clipping for stable training
  - Richer augmentation (Gaussian noise injection)

Requires the candidate caches to exist (run prepare_candidates.py after
generate_dataset.py). Validation metrics: candidate accuracy and how often the
highest-probability candidate is the ground truth.

Example:
    python prepare_candidates.py --split train
    python train.py --config configs/default.json --epochs 60 --variant global
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.dataset import CandidateDataset, collate_candidates
from model.ranker import Ranker


class _Subset(Dataset):
    def __init__(self, base, idxs):
        self.base = base
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        return self.base[self.idxs[i]]


class _PrecompDataset(Dataset):
    """Dataset over the pre-materialised ranker_inputs.npz (fast training).
    Enhanced augmentation: Gaussian noise, random brightness/gamma, flips.
    """
    def __init__(self, npz_path: str, aug: bool = True):
        d = np.load(npz_path)
        self.X = d["X"]
        self.F = d["F"]
        self.y = d["y"]
        self.mask = d["mask"]
        self.aug = aug

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i].copy()
        F = self.F[i].copy()
        y = self.y[i].copy()
        if self.aug:
            rnd = random.Random(i + int(time.time() * 1000) % 10000)
            # Spatial flips (valid because search+reference both flip)
            if rnd.random() < 0.5:
                X = X[:, :, ::-1].copy()
            if rnd.random() < 0.5:
                X = X[:, :, :, ::-1].copy()
            # Brightness / contrast on search channel with improved augmentation
            ch0 = X[:, 0] * rnd.uniform(0.80, 1.25) + rnd.uniform(-0.08, 0.08)
            X[:, 0] = np.clip(ch0, -5.0, 5.0)
            # Gaussian noise injection (mimics severe SEM noise) - more aggressive
            if rnd.random() < 0.5:
                sigma = rnd.uniform(0.08, 0.4)
                noise = np.random.default_rng().normal(0.0, sigma, X[:, 0].shape).astype(np.float32)
                X[:, 0] = np.clip(X[:, 0] + noise, -5.0, 5.0)
            # Random gamma (non-linear intensity mapping) on search channel
            if rnd.random() < 0.4:
                gamma = rnd.uniform(0.7, 1.3)
                # Apply gamma in [0, 1] space then re-z-score
                ch = X[:, 0]
                mn, mx_ = ch.min(), ch.max()
                if mx_ > mn:
                    ch_n = (ch - mn) / (mx_ - mn + 1e-6)
                    ch_n = np.power(np.clip(ch_n, 0.0, 1.0), gamma)
                    ch = ch_n * (mx_ - mn) + mn
                X[:, 0] = ch
            # Additional salt-and-pepper noise on search
            if rnd.random() < 0.2:
                prob = rnd.uniform(0.005, 0.02)
                mask_sp = np.random.uniform(0, 1, X[:, 0].shape) < prob
                X[:, 0][mask_sp] = np.random.choice([-5.0, 5.0], size=int(mask_sp.sum()))
        return (torch.from_numpy(X), torch.from_numpy(F),
                torch.from_numpy(y))


def focal_loss(logits: torch.Tensor, labels: torch.Tensor,
               pos_weight: torch.Tensor | None = None,
               gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """Binary Focal Loss.
    Down-weights easy negatives (well-classified) and focuses training on
    hard positives/negatives. Critical for imbalanced candidate sets where
    most candidates are true negatives.
    """
    # Standard BCE term (unweighted for focal re-weighting)
    bce = F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * labels + (1 - p) * (1 - labels)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    focal_weight = alpha_t * (1.0 - p_t) ** gamma
    return (focal_weight * bce).mean()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N cached samples")
    p.add_argument("--variant", default="local", choices=["local", "global"])
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="linear LR warm-up epochs before cosine decay")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=0.25)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="gradient norm clipping (0 = disabled)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = json.load(open(args.config))
    dcfg, tcfg = cfg["dataset"], cfg["train"]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    epochs = args.epochs or tcfg["epochs"]
    batch = args.batch_size or tcfg["batch_size"]
    lr = args.lr or tcfg["lr"]
    k = tcfg["candidate_k"]

    split_root = os.path.join(dcfg["root"], "train")
    precomp = os.path.join(split_root, f"ranker_inputs_{args.variant}.npz")
    if os.path.exists(precomp):
        ds = _PrecompDataset(precomp, aug=True)
        print(f"using precomputed inputs: {precomp}")
    else:
        if not os.path.isdir(os.path.join(split_root, "candidates")):
            raise SystemExit("No candidate caches found. Run prepare_candidates.py first.")
        ds = CandidateDataset(split_root, aug=True,
                              pos_margin_px=tcfg["pos_margin_px"], k=k)
    if args.limit:
        ds = _Subset(ds, list(range(min(args.limit, len(ds)))))
    n_val = max(1, int(len(ds) * tcfg.get("val_split", 0.1)))
    n_train = len(ds) - n_val
    idxs = list(range(len(ds)))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idxs)
    train_ds = _Subset(ds, idxs[:n_train])
    val_ds = _Subset(ds, idxs[n_train:])

    collate = lambda b: collate_candidates(b, k)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            collate_fn=collate, num_workers=0)

    in_ch = 3 if args.variant == "global" else 2
    model = Ranker(n_feat=1 + len(cfg["infer"]["scales"]), in_ch=in_ch)
    model.to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=tcfg["weight_decay"])

    # Cosine annealing LR scheduler
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs - args.warmup_epochs))

    # Compute class balance for pos_weight
    n_pos = 0
    n_tot = 0
    if isinstance(ds, _PrecompDataset):
        y_all = ds.y[np.asarray(train_ds.idxs) if hasattr(train_ds, "idxs") else ...]
        m = y_all > -1
        n_pos = int((y_all[m] > 0.5).sum())
        n_tot = int(m.sum())
    else:
        for i in range(len(train_ds)):
            d = np.load(train_ds.base.npz_paths[train_ds.idxs[i]], allow_pickle=True)
            lab = np.asarray([c["label"] for c in d["cands"]])
            n_pos += int(lab.sum())
            n_tot += len(lab)
    pos_rate = n_pos / max(n_tot, 1)
    pos_weight = torch.tensor([(1.0 - pos_rate) / max(pos_rate, 1e-3)]).to(args.device)
    print(f"positive rate {pos_rate:.3f} -> pos_weight {pos_weight.item():.2f}")

    os.makedirs(os.path.dirname(tcfg["checkpoint"]) or ".", exist_ok=True)
    grad_accum_steps = tcfg.get("gradient_accumulation_steps", 1)
    print(f"train={len(train_ds)} val={len(val_ds)} cands/img={k} "
          f"batch={batch} epochs={epochs} lr={lr} "
          f"warmup={args.warmup_epochs} focal(gamma={args.focal_gamma},alpha={args.focal_alpha}) "
          f"grad_accum={grad_accum_steps}")

    best_val = -1.0
    for epoch in range(epochs):
        # --- LR warm-up: linearly ramp from lr/10 to lr ---
        if epoch < args.warmup_epochs:
            warmup_factor = (epoch + 1) / max(args.warmup_epochs, 1)
            for pg in opt.param_groups:
                pg["lr"] = lr * warmup_factor
        elif epoch == args.warmup_epochs:
            # Reset to base lr before cosine takes over
            for pg in opt.param_groups:
                pg["lr"] = lr

        model.train()
        t0 = time.time()
        tot = 0.0
        steps = 0
        tr_hits = []
        accum_step = 0
        for batch_idx, (X, F, y, mask) in enumerate(train_loader):
            X, F, y = X.to(args.device), F.to(args.device), y.to(args.device)
            logits = model(X.flatten(0, 1), F.flatten(0, 1))
            labels = y.flatten()
            valid = mask.flatten()
            loss = focal_loss(logits[valid], labels[valid],
                              pos_weight=pos_weight,
                              gamma=args.focal_gamma,
                              alpha=args.focal_alpha)
            # Gradient accumulation
            loss = loss / grad_accum_steps
            loss.backward()
            accum_step += 1
            
            if accum_step % grad_accum_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                opt.zero_grad()
                steps += 1
            
            tot += float(loss.item()) * grad_accum_steps
            with torch.no_grad():
                prob = torch.sigmoid(logits).view(X.shape[0], k)
                for i in range(X.shape[0]):
                    m = mask[i]
                    py = y[i][m]
                    pp = prob[i][m]
                    pos_ids = torch.nonzero(py > 0.5).flatten().tolist()
                    if pos_ids:
                        tr_hits.append(int(pp.argmax() in pos_ids))

        # Advance cosine LR only after warm-up
        if epoch >= args.warmup_epochs:
            cosine_sched.step()

        model.eval()
        with torch.no_grad():
            accs, hits, hits_pos = [], [], []
            for X, F, y, mask in val_loader:
                X, F, y = X.to(args.device), F.to(args.device), y.to(args.device)
                logits = model(X.flatten(0, 1), F.flatten(0, 1)).view(X.shape[0], k)
                prob = torch.sigmoid(logits)
                for i in range(X.shape[0]):
                    m = mask[i]
                    py = y[i][m]
                    pp = prob[i][m]
                    accs.append(((pp > 0.5) == (py > 0.5)).float().mean().item())
                    pos_ids = torch.nonzero(py > 0.5).flatten().tolist()
                    hits.append(int(pp.argmax() in pos_ids))
                    if pos_ids:
                        hits_pos.append(int(pp.argmax() in pos_ids))
            acc = float(np.mean(accs))
            hit = float(np.mean(hits))
            hit_pos = float(np.mean(hits_pos)) if hits_pos else float("nan")
            tr_hit = float(np.mean(tr_hits)) if tr_hits else float("nan")
            cur_lr = opt.param_groups[0]["lr"]
            print(f"[{epoch+1}/{epochs}] loss={tot/max(steps,1):.4f} lr={cur_lr:.5f} "
                  f"train_top1={tr_hit*100:.1f}% "
                  f"val_acc={acc*100:.1f}% val_top1={hit*100:.1f}% "
                  f"val_top1_pos={hit_pos*100:.1f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if hit > best_val:
            best_val = hit
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg,
                "epoch": epoch,
                "val_acc": acc,
                "val_top1_hit": hit,
            }, tcfg["checkpoint"])
            print(f"  ✓ saved checkpoint -> {tcfg['checkpoint']} (val_top1={hit*100:.1f}%)")

    print(f"\ndone. best val top1-hit = {best_val*100:.1f}%")


if __name__ == "__main__":
    main()
