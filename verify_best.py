# ============================================================
# 验证最佳 MLP++ 配置：noBN + GELU + Res + multitask
# 预期 AUC > 0.6958 (配置G) 或至少 > 0.6944 (baseline)
# ============================================================
import os, sys
import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

from configs.config import (DATA_PROCESSED, EMB_DIM, HIDDEN, BATCH_SIZE,
                            LR, L2, DROPOUT, SEED)
from src.models.models import OptimizedMLP, MLPEmbedding
from main import load_data, make_loader, train_torch_model

torch.manual_seed(SEED); np.random.seed(SEED)

d, vocab = load_data()
sparse_names = list(d["sparse_cols"])
dense_dim = d["dense"].shape[1]
train_loader = make_loader(d, train=True)
test_loader = make_loader(d, train=False)
vocab_sizes = {n: vocab[n] for n in sparse_names}

class MLPCls(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.head = nn.Linear(base.out_dim, 1)
    def forward(self, sparse, dense, sparse_names, seq_sid=None, seq_dt=None):
        return {"ctr": self.head(self.base(sparse, dense, sparse_names)).squeeze(-1)}

def run(name, model, multitask, epochs=12, lr=LR, dropout=DROPOUT, patience=5):
    res = train_torch_model(model, train_loader, test_loader, sparse_names,
                            multitask=multitask, epochs=epochs,
                            use_scheduler=False, grad_clip=5.0,
                            lr=lr, patience=patience)
    print(f"  {name:40s} AUC={