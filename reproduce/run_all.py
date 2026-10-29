# ============================================================
# 批量运行：FM / WideDeep / DeepFM，多 seed，结果汇总
# 用法: python run_all.py [--epochs 5] [--seeds 42,2024,2025]
# ============================================================
import os
import sys
import time
import argparse
import subprocess
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

PY = os.path.join(_PROJECT_ROOT, ".venv", "Scripts", "python.exe")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "reproduce", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", default="42,2024,2025")
    parser.add_argument("--models", default="FM,WideDeep,DeepFM")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    models = args.models.split(",")
    all_results = []

    for model in models:
        for seed in seeds:
            print(f"\n{'='*60}\n[RUN] model={model} seed={seed}\n{'='*60}")
            t0 = time.time()
            cmd = [PY, os.path.join(_PROJECT_ROOT, "reproduce", "run_repro.py"),
                   "--model", model, "--seed", str(seed), "--epochs", str(args.epochs)]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
            print(r.stdout[-1200:] if len(r.stdout) > 1200 else r.stdout)
            if r.returncode != 0:
                print("STDERR:", r.stderr[-500:])
            f = os.path.join(RESULTS_DIR, f"repro_{model}_seed{seed}.csv")
            if os.path.exists(f):
                res = pd.read_csv(f)
                res["time_s"] = round(time.time() - t0, 1)
                all_results.append(res)

    if all_results:
        df = pd.concat(all_results, ignore_index=True)
        summary = df.groupby("model").agg(
            auc_mean=("auc", "mean"), auc_std=("auc", "std"),
            logloss_mean=("logloss", "mean"), n_seeds=("seed", "count")
        ).round(4)
        out = os.path.join(RESULTS_DIR, "summary_baselines.csv")
        df.to_csv(out, index=False)
        print("\n" + "=" * 60)
        print("汇总结果 (多 seed 平均):")
        print("=" * 60)
        print(summary.to_string())
        # 论文对照
        paper = {"FM": 0.8376, "WideDeep": 0.8385, "DeepFM": 0.8425}
        print("\n论文报告 (Item=Room, CTR AUC):")
        for m, v in paper.items():
            print(f"  {m}: {v}")
        print(f"\n详细结果已保存: {out}")


if __name__ == "__main__":
    main()
