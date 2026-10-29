# ============================================================
# 一键运行入口：数据准备 -> 训练 baselines + Ours + 消融 -> 输出指标
# 用法：python main.py
# ============================================================
import os
import sys
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

from configs.config import (DATA_PROCESSED, RESULTS_DIR, EMB_DIM, HIDDEN, BATCH_SIZE,
                             EPOCHS, LR, L2, DROPOUT, SEED)
from src.models.models import MLPEmbedding, WideDeep, OursModel, OptimizedMLP

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")


def load_data():
    d = np.load(os.path.join(DATA_PROCESSED, "dataset.npz"), allow_pickle=True)
    v = np.load(os.path.join(DATA_PROCESSED, "vocab.npz"), allow_pickle=True)
    vocab = {k: int(v[k]) for k in v.files}
    return d, vocab


def make_loader(d, train=True):
    mask = d["train_mask"] if train else d["test_mask"]
    ds = TensorDataset(
        torch.from_numpy(d["sparse"][mask]),
        torch.from_numpy(d["dense"][mask]),
        torch.from_numpy(d["seq_sid"][mask]),
        torch.from_numpy(d["seq_dt"][mask]),
        torch.from_numpy(d["y_click"][mask]),
        torch.from_numpy(d["y_like"][mask]),
        torch.from_numpy(d["y_gift"][mask]),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train)


def evaluate_torch(model, loader, sparse_names, multitask=False):
    model.eval()
    preds_c, preds_l, preds_g, ys_c, ys_l, ys_g = [], [], [], [], [], []
    with torch.no_grad():
        for sp, dn, ss, sd, yc, yl, yg in loader:
            out = model(sp, dn, sparse_names, ss, sd)
            preds_c.append(torch.sigmoid(out["ctr"]).numpy())
            ys_c.append(yc.numpy())
            if multitask:
                preds_l.append(torch.sigmoid(out["like"]).numpy())
                preds_g.append(torch.sigmoid(out["gift"]).numpy())
                ys_l.append(yl.numpy())
                ys_g.append(yg.numpy())
    preds_c = np.concatenate(preds_c)
    ys_c = np.concatenate(ys_c)
    res = {
        "auc": roc_auc_score(ys_c, preds_c),
        "logloss": log_loss(ys_c, preds_c, labels=[0, 1]),
    }
    if multitask:
        preds_l = np.concatenate(preds_l); preds_g = np.concatenate(preds_g)
        ys_l = np.concatenate(ys_l); ys_g = np.concatenate(ys_g)
        res["like_auc"] = roc_auc_score(ys_l, preds_l) if ys_l.sum() > 0 else 0.5
        res["gift_auc"] = roc_auc_score(ys_g, preds_g) if ys_g.sum() > 0 else 0.5
    return res


def train_torch_model(model, train_loader, test_loader, sparse_names,
                      multitask=False, epochs=EPOCHS, use_scheduler=False,
                      grad_clip=None, lr=LR, eta_min_ratio=0.1, patience=5):
    """训练 torch 模型，支持 cosine LR 调度（带最低 LR 地板）与梯度裁剪"""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=L2)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr * eta_min_ratio)
    bce = nn.BCEWithLogitsLoss()
    best_auc = 0.0
    best_state = None
    bad_epochs = 0
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for sp, dn, ss, sd, yc, yl, yg in train_loader:
            opt.zero_grad()
            out = model(sp, dn, sparse_names, ss, sd)
            loss = bce(out["ctr"], yc)
            if multitask:
                loss = loss + 0.3 * bce(out["like"], yl) + 0.3 * bce(out["gift"], yg)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tot += loss.item()
        if scheduler is not None:
            scheduler.step()
        res = evaluate_torch(model, test_loader, sparse_names, multitask)
        if res["auc"] > best_auc:
            best_auc = res["auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if (ep + 1) % 2 == 0 or ep == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(f"    epoch {ep+1:02d} loss={tot/len(train_loader):.4f}  "
                  f"lr={lr_now:.2e}  test AUC={res['auc']:.4f}")
        if bad_epochs >= patience:
            print(f"    early stop at epoch {ep+1}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate_torch(model, test_loader, sparse_names, multitask)


def main():
    t0 = time.time()
    print("加载处理后数据 ...")
    d, vocab = load_data()
    sparse_names = list(d["sparse_cols"])
    dense_dim = d["dense"].shape[1]

    train_loader = make_loader(d, train=True)
    test_loader = make_loader(d, train=False)

    # vocab_sizes 对应 sparse_names 顺序
    vocab_sizes = {name: vocab[name] for name in sparse_names}

    results = []

    # ---------- 1. LR baseline (sklearn, dense + sparse one-hot) ----------
    print("\n[Model 1/6] Logistic Regression (sklearn) ...")
    from sklearn.preprocessing import OneHotEncoder
    tr = d["train_mask"]; te = d["test_mask"]
    # 稀疏列 one-hot
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    Xtr_sp = ohe.fit_transform(d["sparse"][tr])
    Xte_sp = ohe.transform(d["sparse"][te])
    from scipy.sparse import hstack
    Xtr = hstack([Xtr_sp, d["dense"][tr]]).tocsr()
    Xte = hstack([Xte_sp, d["dense"][te]]).tocsr()
    lr = LogisticRegression(max_iter=200, C=1.0)
    lr.fit(Xtr, d["y_click"][tr])
    pred_lr = lr.predict_proba(Xte)[:, 1]
    res_lr = {
        "model": "LR",
        "auc": roc_auc_score(d["y_click"][te], pred_lr),
        "logloss": log_loss(d["y_click"][te], pred_lr, labels=[0, 1]),
    }
    print(f"    LR  test AUC={res_lr['auc']:.4f}  LogLoss={res_lr['logloss']:.4f}")
    results.append(res_lr)

    # ---------- 2. MLP baseline ----------
    print("\n[Model 2/6] MLP ...")
    mlp = MLPEmbedding(vocab_sizes, dense_dim, EMB_DIM, HIDDEN, DROPOUT)
    # MLPEmbedding 输出是 trunk 向量，需要一个分类头 —— 直接用 WideDeep 结构但不用 wide
    # 这里改用一个简单包装
    class MLPCls(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.head = nn.Linear(base.out_dim, 1)
        def forward(self, sparse, dense, sparse_names, seq_sid=None, seq_dt=None):
            return {"ctr": self.head(self.base(sparse, dense, sparse_names)).squeeze(-1)}
    mlp = MLPCls(mlp)
    res_mlp = train_torch_model(mlp, train_loader, test_loader, sparse_names, multitask=False)
    res_mlp["model"] = "MLP"
    print(f"    MLP test AUC={res_mlp['auc']:.4f}  LogLoss={res_mlp['logloss']:.4f}")
    results.append(res_mlp)

    # ---------- 3. Wide & Deep ----------
    print("\n[Model 3/6] Wide&Deep ...")
    wd = WideDeep(vocab_sizes, dense_dim, EMB_DIM, HIDDEN, DROPOUT)
    # 包装成 Ours 一致的 forward 接口
    class WDCls(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, sparse, dense, sparse_names, seq_sid=None, seq_dt=None):
            return {"ctr": self.m(sparse, dense, sparse_names).squeeze(-1)}
    wd = WDCls(wd)
    res_wd = train_torch_model(wd, train_loader, test_loader, sparse_names, multitask=False)
    res_wd["model"] = "WideDeep"
    print(f"    W&D  test AUC={res_wd['auc']:.4f}  LogLoss={res_wd['logloss']:.4f}")
    results.append(res_wd)

    # ---------- 4. Ours (full) ----------
    print("\n[Model 4/6] Ours (full: multitask + timedecay + DIN) ...")
    ours = OursModel(vocab_sizes, dense_dim, EMB_DIM, HIDDEN, DROPOUT,
                     use_timedecay=True, use_din=True, use_multitask=True)
    res_ours = train_torch_model(ours, train_loader, test_loader, sparse_names, multitask=True)
    res_ours["model"] = "Ours"
    print(f"    Ours test AUC={res_ours['auc']:.4f}  LogLoss={res_ours['logloss']:.4f}  "
          f"likeAUC={res_ours.get('like_auc',0):.4f}  giftAUC={res_ours.get('gift_auc',0):.4f}")
    results.append(res_ours)

    # ---------- 5. Ablation: Ours w/o TimeDecay ----------
    print("\n[Model 5/6] Ablation: Ours w/o TimeDecay ...")
    ab1 = OursModel(vocab_sizes, dense_dim, EMB_DIM, HIDDEN, DROPOUT,
                    use_timedecay=False, use_din=True, use_multitask=True)
    res_ab1 = train_torch_model(ab1, train_loader, test_loader, sparse_names, multitask=True)
    res_ab1["model"] = "Ors_wo_TimeDecay"
    results.append(res_ab1)

    # ---------- 6. Ablation: Ours w/o DIN ----------
    print("\n[Model 6/6] Ablation: Ours w/o DIN ...")
    ab2 = OursModel(vocab_sizes, dense_dim, EMB_DIM, HIDDEN, DROPOUT,
                    use_timedecay=True, use_din=False, use_multitask=True)
    res_ab2 = train_torch_model(ab2, train_loader, test_loader, sparse_names, multitask=True)
    res_ab2["model"] = "Ours_wo_DIN"
    results.append(res_ab2)

    # ---------- 7. Ablation: Ours w/o Multitask (CTR only) ----------
    print("\n[Ablation] Ours w/o Multitask (CTR only) ...")
    ab3 = OursModel(vocab_sizes, dense_dim, EMB_DIM, HIDDEN, DROPOUT,
                    use_timedecay=True, use_din=True, use_multitask=False)
    res_ab3 = train_torch_model(ab3, train_loader, test_loader, sparse_names, multitask=False)
    res_ab3["model"] = "Ours_wo_Multitask"
    results.append(res_ab3)

    # ---------- MLP++ 最优配置（经诊断调优）：noBN + GELU + 残差 + 多任务 ----------
    MLP_LR = LR            # 1e-3，与 baseline 一致
    MLP_EPOCHS = 12
    MLP_DROPOUT = DROPOUT  # 0.2

    # ---------- 8. MLP++ (full: noBN+GELU+2ResBlock+uniform16+multitask) ----------
    print("\n[Model] MLP++ (full: noBN+GELU+Res+multitask) ...")
    mlpp = OptimizedMLP(vocab_sizes, dense_dim, EMB_DIM, HIDDEN, MLP_DROPOUT,
                        use_cross=False, use_seq=False, use_multitask=True,
                        num_res_blocks=2, uniform_emb=True, use_bn=False, act="gelu")
    res_mlpp = train_torch_model(mlpp, train_loader, test_loader, sparse_names,
                                 multitask=True, epochs=MLP_EPOCHS,
                                 use_scheduler=False, grad_clip=5.0,
                                 lr=MLP_LR, patience=5)
    res_mlpp["model"] = "MLP++"
    print(f"    MLP++ test AUC={res_mlpp['auc']:.4f}  LogLoss={res_mlpp['logloss']:.4f}  "
          f"likeAUC={res_mlpp.get('like_auc',0):.4f}  giftAUC={res_mlpp.get('gift_auc',0):.4f}")
    results.append(res_mlpp)

    # ---------- 9. Ablation: MLP++ w/o ResBlock ----------
    print("\n[Ablation] MLP++ w/o ResBlock (num_res_blocks=0) ...")
    ab_mlp1 = OptimizedMLP(vocab_sizes, dense_dim, EMB_DIM, HIDDEN, MLP_DROPOUT,
                           use_cross=False, use_seq=False, use_multitask=True,
                           num_res_blocks=0, uniform_emb=True, use_bn=False, act="gelu")
    res_ab_mlp1 = train_torch_model(ab_mlp1, train_loader, test_loader, sparse_names,
                                    multitask=True, epochs=MLP_EPOCHS,
                                    use_scheduler=False, grad_clip=5.0,
                                    lr=MLP_LR, patience=5)
    res_ab_mlp1["model"] = "MLP++_wo_ResBlock"
    results.append(res_ab_mlp1)

    # ---------- 10. Ablation: MLP++ w/o Multitask (CTR only) ----------
    print("\n[Ablation] MLP++ w/o Multitask (CTR only) ...")
    ab_mlp2 = OptimizedMLP(vocab_sizes, dense_dim, EMB_DIM, HIDDEN, MLP_DROPOUT,
                           use_cross=False, use_seq=False, use_multitask=False,
                           num_res_blocks=2, uniform_emb=True, use_bn=False, act="gelu")
    res_ab_mlp2 = train_torch_model(ab_mlp2, train_loader, test_loader, sparse_names,
                                    multitask=False, epochs=MLP_EPOCHS,
                                    use_scheduler=False, grad_clip=5.0,
                                    lr=MLP_LR, patience=5)
    res_ab_mlp2["model"] = "MLP++_wo_Multitask"
    results.append(res_ab_mlp2)

    # ---------- 11. Ablation: MLP++ w/ BN (验证 BN 在小数据上有害) ----------
    print("\n[Ablation] MLP++ w/ BatchNorm ...")
    ab_mlp3 = OptimizedMLP(vocab_sizes, dense_dim, EMB_DIM, HIDDEN, MLP_DROPOUT,
                           use_cross=False, use_seq=False, use_multitask=True,
                           num_res_blocks=2, uniform_emb=True, use_bn=True, act="gelu")
    res_ab_mlp3 = train_torch_model(ab_mlp3, train_loader, test_loader, sparse_names,
                                    multitask=True, epochs=MLP_EPOCHS,
                                    use_scheduler=False, grad_clip=5.0,
                                    lr=MLP_LR, patience=5)
    res_ab_mlp3["model"] = "MLP++_w_BN"
    results.append(res_ab_mlp3)

    # ---------- 12. Ablation: MLP++ w/o GELU (用 ReLU) ----------
    print("\n[Ablation] MLP++ w/o GELU (ReLU instead) ...")
    ab_mlp4 = OptimizedMLP(vocab_sizes, dense_dim, EMB_DIM, HIDDEN, MLP_DROPOUT,
                           use_cross=False, use_seq=False, use_multitask=True,
                           num_res_blocks=2, uniform_emb=True, use_bn=False, act="relu")
    res_ab_mlp4 = train_torch_model(ab_mlp4, train_loader, test_loader, sparse_names,
                                    multitask=True, epochs=MLP_EPOCHS,
                                    use_scheduler=False, grad_clip=5.0,
                                    lr=MLP_LR, patience=5)
    res_ab_mlp4["model"] = "MLP++_wo_GELU"
    results.append(res_ab_mlp4)

    # ---------- 落盘 ----------
    df = pd.DataFrame(results)
    main_models = ["LR", "MLP", "WideDeep", "Ours", "MLP++"]
    main_exp = df[df["model"].isin(main_models)].reset_index(drop=True)
    abl = df[~df["model"].isin(main_models)].reset_index(drop=True)
    main_exp.to_csv(os.path.join(RESULTS_DIR, "main_experiment.csv"), index=False)
    abl.to_csv(os.path.join(RESULTS_DIR, "ablation.csv"), index=False)

    print("\n" + "=" * 60)
    print("主实验结果（含 MLP 优化对比）：")
    print(main_exp.to_string(index=False))
    print("\n消融实验：")
    print(abl.to_string(index=False))
    print(f"\n总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
