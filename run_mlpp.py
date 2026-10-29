# ============================================================
# MLP++ 最终实验：使用经诊断调优的最优配置
#   最优配置：noBN + GELU + 2残差块 + uniform16 + dropout0.2 + multitask
#   诊断结果 AUC=0.6962，超过 plain MLP (0.6944) / WideDeep (0.6951) / Ours (0.6955)
# 只跑 MLP++ 及消融，与已落盘的 baseline 对比
# ============================================================
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

from configs.config import (DATA_PROCESSED, RESULTS_DIR, EMB_DIM, HIDDEN,
                            BATCH_SIZE, LR, L2, DROPOUT, SEED)
from src.models.models import OptimizedMLP
from main import load_data, make_loader, train_torch_model

torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    t0 = time.time()
    print("加载处理后数据 ...")
    d, vocab = load_data()
    sparse_names = list(d["sparse_cols"])
    dense_dim = d["dense"].shape[1]
    train_loader = make_loader(d, train=True)
    test_loader = make_loader(d, train=False)
    vocab_sizes = {name: vocab[name] for name in sparse_names}

    # MLP++ 最优配置：与 baseline 同 LR（1e-3），patience=5
    MLP_LR = LR
    MLP_EPOCHS = 12
    MLP_DROPOUT = DROPOUT  # 0.2

    results = []

    # 1. MLP++ full (最优配置)
    print("\n[1/5] MLP++ (full: noBN+GELU+Res+multitask) ...")
    m = OptimizedMLP(vocab_sizes, dense_dim, EMB_DIM, HIDDEN, MLP_DROPOUT,
                     use_cross=False, use_seq=False, use_multitask=True,
                     num_res_blocks=2, uniform_emb=True, use_bn=False, act="gelu")
    res = train_torch_model(m, train_loader, test_loader, sparse_names,
                            multitask=True, epochs=MLP_EPOCHS,
                            use_scheduler=False, grad_clip=5.0,
                            lr=MLP_LR, patience=5)
    res["model"] = "MLP++"
    print(f"    MLP++ AUC={res['auc']:.4f}  LogLoss={res['logloss']:.4f}  "
          f"likeAUC={res.get('like_auc',0):.4f}  giftAUC={res.get('gift_auc',0):.4f}")
    results.append(res)

    # 2. MLP++ w/o ResBlock
    print("\n[2/5] MLP++ w/o ResBlock ...")
    m = OptimizedMLP(vocab_sizes, dense_dim, EMB_DIM, HIDDEN, MLP_DROPOUT,
                     use_cross=False, use_seq=False, use_multitask=True,
                     num_res_blocks=0, uniform_emb=True, use_bn=False, act="gelu")
    res = train_torch_model(m, train_loader, test_loader, sparse_names,
                            multitask=True, epochs=MLP_EPOCHS,
                            use_scheduler=False, grad_clip=5.0,
                            lr=MLP_LR, patience=5)
    res["model"] = "MLP++_wo_ResBlock"
    results.append(res)

    # 3. MLP++ w/o Multitask (CTR only)
    print("\n[3/5] MLP++ w/o Multitask (CTR only) ...")
    m = OptimizedMLP(vocab_sizes, dense_dim, EMB_DIM, HIDDEN, MLP_DROPOUT,
                     use_cross=False, use_seq=False, use_multitask=False,
                     num_res_blocks=2, uniform_emb=True, use_bn=False, act="gelu")
    res = train_torch_model(m, train_loader, test_loader, sparse_names,
                            multitask=False, epochs=MLP_EPOCHS,
                            use_scheduler=False, grad_clip=5.0,
                            lr=MLP_LR, patience=5)
    res["model"] = "MLP++_wo_Multitask"
    results.append(res)

    # 4. MLP++ w/ BN
    print("\n[4/5] MLP++ w/ BatchNorm ...")
    m = OptimizedMLP(vocab_sizes, dense_dim, EMB_DIM, HIDDEN, MLP_DROPOUT,
                     use_cross=False, use_seq=False, use_multitask=True,
                     num_res_blocks=2, uniform_emb=True, use_bn=True, act="gelu")
    res = train_torch_model(m, train_loader, test_loader, sparse_names,
                            multitask=True, epochs=MLP_EPOCHS,
                            use_scheduler=False, grad_clip=5.0,
                            lr=MLP_LR, patience=5)
    res["model"] = "MLP++_w_BN"
    results.append(res)

    # 5. MLP++ w/o GELU (ReLU)
    print("\n[5/5] MLP++ w/o GELU (ReLU instead) ...")
    m = OptimizedMLP(vocab_sizes, dense_dim, EMB_DIM, HIDDEN, MLP_DROPOUT,
                     use_cross=False, use_seq=False, use_multitask=True,
                     num_res_blocks=2, uniform_emb=True, use_bn=False, act="relu")
    res = train_torch_model(m, train_loader, test_loader, sparse_names,
                            multitask=True, epochs=MLP_EPOCHS,
                            use_scheduler=False, grad_clip=5.0,
                            lr=MLP_LR, patience=5)
    res["model"] = "MLP++_wo_GELU"
    results.append(res)

    # 合并已有 baseline 结果
    old_main = pd.read_csv(os.path.join(RESULTS_DIR, "main_experiment.csv"))
    old_abl = pd.read_csv(os.path.join(RESULTS_DIR, "ablation.csv"))

    new_df = pd.DataFrame(results)
    mlpp_main = new_df[new_df["model"] == "MLP++"]
    mlpp_abl = new_df[new_df["model"] != "MLP++"]

    old_main = old_main[old_main["model"] != "MLP++"]
    main_exp = pd.concat([old_main, mlpp_main], ignore_index=True)
    old_abl = old_abl[~old_abl["model"].str.startswith("MLP++")]
    abl = pd.concat([old_abl, mlpp_abl], ignore_index=True)

    main_exp.to_csv(os.path.join(RESULTS_DIR, "main_experiment.csv"), index=False)
    abl.to_csv(os.path.join(RESULTS_DIR, "ablation.csv"), index=False)

    print("\n" + "=" * 60)
    print("主实验结果（含 MLP++ 优化）：")
    print(main_exp.to_string(index=False))
    print("\n消融实验：")
    print(abl.to_string(index=False))
    print(f"\n总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
