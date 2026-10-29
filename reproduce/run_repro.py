# ============================================================
# 复现主入口 —— 对齐 ReChorus CTRRunner
# 协议: BCE loss + AUC/Logloss 评估
# 用法: python run_repro.py --model FM|WideDeep|DeepFM [--seed 42]
# ============================================================
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
from reproduce.models_repro import FMModel, WideDeepModel, DeepFMModel
from reproduce.models_optimized import FIDeepFMModel

DATA_DIR = os.path.join(_PROJECT_ROOT, "reproduce", "data_repro")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "reproduce", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class CTRDataset(Dataset):
    def __init__(self, df, feature_max, context_features):
        df = df.reset_index(drop=True)
        self.n = len(df)
        self.context_features = context_features
        # 预转 numpy: 离散 long / 连续 float
        self.cat_cols = [f for f in context_features if f.endswith("_id") or f.endswith("_c")]
        self.dense_cols = [f for f in context_features if not (f.endswith("_id") or f.endswith("_c"))]
        self.cat_data = {f: df[f].values.astype(np.int64) for f in self.cat_cols}
        self.dense_data = {f: df[f].values.astype(np.float32) for f in self.dense_cols}
        self.labels = df["label"].values.astype(np.float32)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        feed = {"label": self.labels[idx]}
        for f in self.cat_cols:
            feed[f] = self.cat_data[f][idx]
        for f in self.dense_cols:
            feed[f] = self.dense_data[f][idx]
        return feed


def collate_fn(batch):
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if key == "label":
            out[key] = torch.tensor(vals, dtype=torch.float32)
        elif any(key.endswith(s) for s in ("_id", "_c")):
            out[key] = torch.tensor(vals, dtype=torch.long)
        else:
            out[key] = torch.tensor(vals, dtype=torch.float32)
    return out


def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    dev_df = pd.read_csv(os.path.join(DATA_DIR, "dev.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    item_meta = pd.read_csv(os.path.join(DATA_DIR, "item_meta.csv"))
    user_meta = pd.read_csv(os.path.join(DATA_DIR, "user_meta.csv"))

    # 合并侧信息到每个 split
    def merge_meta(df):
        df = df.merge(item_meta, on="item_id", how="left")
        df = df.merge(user_meta, on="user_id", how="left")
        return df.fillna(0)

    train_df = merge_meta(train_df)
    dev_df = merge_meta(dev_df)
    test_df = merge_meta(test_df)

    # 确定 context_features (i_ / u_ 前缀列)
    context_features = sorted(
        [c for c in train_df.columns if c.startswith("i_") or c.startswith("u_")]
    )
    # 确保 user_id / item_id 也是特征
    if "user_id" not in context_features:
        context_features = ["user_id", "item_id"] + context_features

    # dense 特征标准化 (用 train 的统计量，防数据泄漏)
    dense_cols = [f for f in context_features if not (f.endswith("_id") or f.endswith("_c"))]
    dense_stats = {}
    for c in dense_cols:
        mu = train_df[c].mean()
        sd = train_df[c].std() + 1e-8
        dense_stats[c] = (float(mu), float(sd))
        train_df[c] = (train_df[c] - mu) / sd
        dev_df[c] = (dev_df[c] - mu) / sd
        test_df[c] = (test_df[c] - mu) / sd
    print(f"dense 特征已标准化: {dense_cols}")

    # 计算 feature_max
    feature_max = {}
    for f in context_features:
        if f.endswith("_id") or f.endswith("_c"):
            mx = int(pd.concat([train_df[f], dev_df[f], test_df[f]]).max()) + 1
            feature_max[f] = max(mx, 2)
        else:
            feature_max[f] = 1  # dense 用 Linear, 不查表

    print(f"特征数: {len(context_features)}, feature_max: {feature_max}")
    return train_df, dev_df, test_df, feature_max, context_features


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            preds.append(torch.sigmoid(out["logit"]).cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    auc = roc_auc_score(labels, preds)
    ll = log_loss(labels, np.clip(preds, 1e-7, 1 - 1e-7), labels=[0, 1])
    return auc, ll


def train(model_name, seed=42, epochs=20, lr=1e-3, batch_size=2048, emb_size=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Reproduce] model={model_name} seed={seed} device={device}")

    train_df, dev_df, test_df, feature_max, context_features = load_data()
    print(f"train={len(train_df)} dev={len(dev_df)} test={len(test_df)}")

    train_ds = CTRDataset(train_df, feature_max, context_features)
    test_ds = CTRDataset(test_df, feature_max, context_features)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    if model_name == "FM":
        model = FMModel(feature_max, context_features, emb_size).to(device)
    elif model_name == "WideDeep":
        model = WideDeepModel(feature_max, context_features, emb_size, layers=(64,)).to(device)
    elif model_name == "DeepFM":
        model = DeepFMModel(feature_max, context_features, emb_size, layers=(64,)).to(device)
    elif model_name == "FI-DeepFM":
        model = FIDeepFMModel(feature_max, context_features, emb_size, layers=(64,),
                              dropout=0.1, num_heads=2, num_interact_layers=1).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            out = model(batch)
            loss = bce(out["logit"], batch["label"])
            loss.backward()
            opt.step()
            tot += loss.item()
        if (ep + 1) % 2 == 0 or ep == 0:
            auc, ll = evaluate(model, test_loader, device)
            flag = " *" if auc > best_auc else ""
            print(f"  epoch {ep+1:02d}  loss={tot/len(train_loader):.4f}  "
                  f"test AUC={auc:.4f}  Logloss={ll:.4f}  ({time.time()-t0:.1f}s){flag}")
            best_auc = max(best_auc, auc)

    auc, ll = evaluate(model, test_loader, device)
    print(f"[Final] {model_name}: AUC={auc:.4f}  Logloss={ll:.4f}")

    # 论文对照 (Item=Room 时的报告值)
    paper_ref = {
        "FM": (0.8376, 0.4810),
        "WideDeep": (0.8385, 0.4162),
        "DeepFM": (0.8425, 0.4130),
    }
    if model_name in paper_ref:
        p_auc, p_ll = paper_ref[model_name]
        print(f"[Paper] {model_name}: AUC={p_auc:.4f}  Logloss={p_ll:.4f}  "
              f"(ΔAUC={auc-p_auc:+.4f})")

    return {"model": model_name, "seed": seed, "auc": auc, "logloss": ll}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="DeepFM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    res = train(args.model, args.seed, args.epochs)
    df = pd.DataFrame([res])
    out = os.path.join(RESULTS_DIR, f"repro_{args.model}_seed{args.seed}.csv")
    df.to_csv(out, index=False)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
