# ============================================================
# 诊断：为什么 reproduce 框架下 baseline AUC < 0.5
# 从实验里抠问题——先确认 baseline 能不能学到信号
# ============================================================
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
from reproduce.run_repro import load_data, CTRDataset, collate_fn
from reproduce.models_repro import DeepFMModel
from torch.utils.data import DataLoader

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df, dev_df, test_df, feature_max, context_features = load_data()

# 关键诊断 1: 只用 user_id 做 group mean 编码，看信号强度
# 如果 user 历史点击率能预测 test，说明数据有信号
print("\n[诊断1] user 历史点击率 vs test label 的相关性")
tr_pos_rate = train_df[train_df.label == 1].groupby("user_id").size() / train_df.groupby("user_id").size()
tr_pos_rate = tr_pos_rate.fillna(0.5)
test_user_rate = test_df["user_id"].map(tr_pos_rate).fillna(0.5).values
auc_user = roc_auc_score(test_df["label"].values, test_user_rate)
print(f"  只用 user 历史点击率预测 test AUC = {auc_user:.4f}")

# 关键诊断 2: item(room) 历史点击率
item_pos_rate = train_df[train_df.label == 1].groupby("item_id").size() / train_df.groupby("item_id").size()
item_pos_rate = item_pos_rate.fillna(0.5)
test_item_rate = test_df["item_id"].map(item_pos_rate).fillna(0.5).values
auc_item = roc_auc_score(test_df["label"].values, test_item_rate)
print(f"  只用 item 历史点击率预测 test AUC = {auc_item:.4f}")

# 关键诊断 3: 训练一个 DeepFM，看 train AUC（能否拟合 train）
print("\n[诊断3] DeepFM 能否拟合 train 集（过拟合能力）")
train_ds = CTRDataset(train_df, feature_max, context_features)
test_ds = CTRDataset(test_df, feature_max, context_features)
train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False, collate_fn=collate_fn)

model = DeepFMModel(feature_max, context_features, emb_size=64, layers=(64,)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
bce = torch.nn.BCEWithLogitsLoss()

for ep in range(5):
    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad()
        out = model(batch)
        loss = bce(out["logit"], batch["label"])
        loss.backward()
        opt.step()

def get_preds(model, loader):
    model.eval()
    ps, ls = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            ps.append(torch.sigmoid(model(batch)["logit"]).cpu().numpy())
            ls.append(batch["label"].cpu().numpy())
    return np.concatenate(ps), np.concatenate(ls)

tr_p, tr_l = get_preds(model, train_loader)
te_p, te_l = get_preds(model, test_loader)
print(f"  train AUC = {roc_auc_score(tr_l, tr_p):.4f}  (若<0.7说明模型/特征有问题)")
print(f"  test  AUC = {roc_auc_score(te_l, te_p):.4f}")

# 诊断 4: 检查是否 logit 符号反了（AUC<0.5 有时是 label 或预测方向问题）
print(f"\n[诊断4] 预测统计: pred mean={te_p.mean():.4f}, label mean={te_l.mean():.4f}")
print(f"  正样本平均预测分 = {te_p[te_l==1].mean():.4f}")
print(f"  负样本平均预测分 = {te_p[te_l==0].mean():.4f}")
print(f"  正>负? {te_p[te_l==1].mean() > te_p[te_l==0].mean()}  (若 False 说明学反了)")
