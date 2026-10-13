"""Appearance backbone (ResNet-50, last-stride 1, BNNeck, 512-D) + controlled ReID training/eval.

Used for (a) the DukeMTMC-reID ablation (CE vs CE+triplet, fixed schedule, several seeds) and
(b) training the vehicle appearance model on CityFlow TRAIN-scene crops only.
Image folders use Market/Duke naming: <pid>_c<cam>_<anything>.jpg
"""
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, Sampler

IMG_EXT = {".jpg", ".jpeg", ".png"}


# ----------------------------------------------------------------------------- data
def parse_name(name):
    parts = Path(name).stem.split("_")
    return int(parts[0]), int(parts[1].lstrip("c").lstrip("C"))


class ReIDFolder(Dataset):
    def __init__(self, folder, transform, relabel=False):
        self.paths = sorted(p for p in Path(folder).iterdir()
                            if p.suffix.lower() in IMG_EXT and not p.name.startswith("-1"))
        self.pids, self.cams = zip(*[parse_name(p.name) for p in self.paths]) if self.paths else ((), ())
        self.pids, self.cams = list(self.pids), list(self.cams)
        self.transform = transform
        self.pid2label = {p: i for i, p in enumerate(sorted(set(self.pids)))}
        self.relabel = relabel

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        pid = self.pids[i]
        return self.transform(img), (self.pid2label[pid] if self.relabel else pid), self.cams[i]


class PKSampler(Sampler):
    """P identities x K instances per batch (standard for triplet training)."""

    def __init__(self, pids, P=16, K=4, seed=0):
        self.index = defaultdict(list)
        for i, p in enumerate(pids):
            self.index[p].append(i)
        self.P, self.K = P, K
        self.rng = random.Random(seed)
        self.n = sum(max(len(v), K) // K * K for v in self.index.values())

    def __iter__(self):
        chunks = defaultdict(list)
        for p, idx in self.index.items():
            idx = list(idx)
            if len(idx) < self.K:
                idx = idx + self.rng.choices(idx, k=self.K - len(idx))
            self.rng.shuffle(idx)
            for s in range(0, len(idx) - self.K + 1, self.K):
                chunks[p].append(idx[s:s + self.K])
        avail = [p for p in chunks if chunks[p]]
        out = []
        while len(avail) >= self.P:
            for p in self.rng.sample(avail, self.P):
                out.extend(chunks[p].pop())
                if not chunks[p]:
                    avail.remove(p)
        return iter(out)

    def __len__(self):
        return self.n


def transforms(h, w, train):
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return T.Compose([T.Resize((h, w)), T.RandomHorizontalFlip(), T.Pad(10), T.RandomCrop((h, w)),
                          T.ColorJitter(0.2, 0.2, 0.15, 0.02), T.ToTensor(), norm,
                          T.RandomErasing(p=0.5)])
    return T.Compose([T.Resize((h, w)), T.ToTensor(), norm])


# ----------------------------------------------------------------------------- model
class ReIDBackbone(nn.Module):
    """ResNet-50 (ImageNet), last stride 1, GAP, BNNeck, 2048->512 projection."""

    def __init__(self, num_classes=0, emb_dim=512, pretrained=True, last_stride=1):
        super().__init__()
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        r = torchvision.models.resnet50(weights=w)
        if last_stride == 1:
            r.layer4[0].conv2.stride = (1, 1)
            r.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(*list(r.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(2048, emb_dim)
        self.bnneck = nn.BatchNorm1d(emb_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(emb_dim, num_classes, bias=False) if num_classes > 0 else None
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        f = self.proj(self.gap(self.base(x)).flatten(1))   # triplet feature (before BN)
        b = self.bnneck(f)                                   # inference / CE feature
        if self.training and self.classifier is not None:
            return f, self.classifier(b)
        return F.normalize(b, dim=1)


def batch_hard_triplet(feat, labels, margin=0.3):
    d = torch.cdist(feat, feat)
    same = labels[:, None] == labels[None, :]
    ap = (d * same.float()).max(1).values
    an = (d + 1e6 * same.float()).min(1).values
    return F.relu(ap - an + margin).mean()


# ----------------------------------------------------------------------------- train / eval
def train_reid(train_dir, out_ckpt, epochs=30, loss="ce_triplet", triplet_weight=1.0, margin=0.3,
               lr=3.5e-4, P=16, K=4, img_h=256, img_w=128, seed=0, workers=4, device="cuda",
               pretrained=True, log=print, eval_fn=None):
    """Fixed schedule (warm-up 10 ep, x0.1 at 40%/70% of epochs). No model selection on test data:
    the LAST epoch is saved and reported. `eval_fn(model, epoch)` is only for monitoring curves."""
    from functools import partial
    from .common import seed_worker, set_seed
    set_seed(seed)
    ds = ReIDFolder(train_dir, transforms(img_h, img_w, True), relabel=True)
    sampler = PKSampler([ds.pid2label[p] for p in ds.pids], P=P, K=K, seed=seed)
    dl = DataLoader(ds, batch_size=P * K, sampler=sampler, num_workers=workers, drop_last=True,
                    pin_memory=True, worker_init_fn=partial(seed_worker, base_seed=seed),
                    persistent_workers=workers > 0)
    model = ReIDBackbone(num_classes=len(ds.pid2label), pretrained=pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    m1, m2 = int(0.4 * epochs), int(0.7 * epochs)

    def lr_at(e):
        if e < 10 and epochs > 20:
            return lr * (0.1 + 0.9 * e / 10)
        return lr * (1.0 if e < m1 else 0.1 if e < m2 else 0.01)

    use_amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    hist = []
    for ep in range(epochs):
        for g in opt.param_groups:
            g["lr"] = lr_at(ep)
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y, _ in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                f, logits = model(x)
                l = ce(logits, y)
                if loss == "ce_triplet":
                    l = l + triplet_weight * batch_hard_triplet(f.float(), y, margin)
            opt.zero_grad(set_to_none=True)
            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()
            tot += l.item()
            n += 1
        rec = {"epoch": ep + 1, "loss": tot / max(n, 1), "lr": lr_at(ep), "sec": time.time() - t0}
        if eval_fn is not None:
            rec.update(eval_fn(model, ep + 1))
        hist.append(rec)
        log(json.dumps(rec))
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_classes": len(ds.pid2label),
                "img_hw": (img_h, img_w)}, out_ckpt)
    return model, hist


def load_backbone(ckpt, device="cuda"):
    c = torch.load(ckpt, map_location=device)
    m = ReIDBackbone(num_classes=c["num_classes"], pretrained=False).to(device)
    m.load_state_dict(c["state_dict"])
    m.eval()
    return m, tuple(c.get("img_hw", (256, 128)))


@torch.no_grad()
def extract_folder(model, folder, img_h, img_w, device="cuda", bs=256, workers=4):
    ds = ReIDFolder(folder, transforms(img_h, img_w, False))
    dl = DataLoader(ds, batch_size=bs, num_workers=workers)
    model.eval()
    feats = []
    for x, _, _ in dl:
        x = x.to(device)
        f = model(x) + model(torch.flip(x, dims=[3]))   # flip test-time augmentation
        feats.append(F.normalize(f, dim=1).cpu())
    return torch.cat(feats).numpy(), np.array(ds.pids), np.array(ds.cams)


def reid_metrics(qf, qp, qc, gf, gp, gc, max_neg=3_000_000, seed=0):
    """Standard CMC/mAP (same pid AND same cam gallery entries are junk) + threshold-free
    verification metrics (ROC-AUC, EER) over ALL positive query-gallery pairs and a fixed random
    sample of <= max_neg negative pairs (memory; the sample is seeded, so results are reproducible)."""
    sim = qf @ gf.T
    r1, aps, pos_s, neg_s = [], [], [], []
    for i in range(len(qp)):
        keep = ~((gp == qp[i]) & (gc == qc[i]))
        s, y = sim[i][keep], (gp[keep] == qp[i])
        if y.sum() == 0:
            continue
        r1.append(float(y[np.argmax(s)]))
        aps.append(average_precision_score(y, s))
        pos_s.append(s[y])
        neg_s.append(s[~y])
    pos, neg = np.concatenate(pos_s), np.concatenate(neg_s)
    n_neg_total = len(neg)
    if len(neg) > max_neg:
        neg = np.random.default_rng(seed).choice(neg, max_neg, replace=False)
    yv = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    sv = np.r_[pos, neg]
    auc = roc_auc_score(yv, sv)
    fpr, tpr, _ = roc_curve(yv, sv)
    eer = float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])
    return {"rank1": float(np.mean(r1)), "mAP": float(np.mean(aps)), "auc": float(auc), "eer": eer,
            "n_pos_pairs": int(len(pos)), "n_neg_pairs": int(n_neg_total)}
