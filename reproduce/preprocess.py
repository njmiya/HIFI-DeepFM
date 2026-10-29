# ============================================================
# 真实数据预处理 —— 严格对齐论文协议
# 论文协议 (KuaiLive SIGIR'26 + ReChorus):
#   1. leave-one-out 切分：每用户最后1条->test, 倒数第2条->dev, 其余->train
#   2. CTR 任务：正样本=click, 负样本=negative.csv (曝光未点击)
#   3. 侧信息：user_meta.csv (u_ 前缀) / item_meta.csv (i_ 前缀)
#   4. 输出格式兼容 ReChorus ContextReader
# ============================================================
import os
import sys
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(_PROJECT_ROOT, "reproduce", "data_repro")
os.makedirs(OUT_DIR, exist_ok=True)

RNG = np.random.RandomState(42)


def load_raw():
    """加载 KuaiLive 原始 CSV，检查 schema 与论文一致"""
    print("[1/5] 加载原始 CSV ...")
    user = pd.read_csv(os.path.join(RAW_DIR, "user.csv"))
    streamer = pd.read_csv(os.path.join(RAW_DIR, "streamer.csv"))
    room = pd.read_csv(os.path.join(RAW_DIR, "room.csv"))
    click = pd.read_csv(os.path.join(RAW_DIR, "click.csv"))
    neg = pd.read_csv(os.path.join(RAW_DIR, "negative.csv"))

    # 论文 schema 检查
    print(f"  user: {user.shape}, streamer: {streamer.shape}, room: {room.shape}")
    print(f"  click: {click.shape}, negative: {neg.shape}")
    assert "user_id" in user.columns and "streamer_id" in streamer.columns
    assert {"user_id", "room_id", "timestamp"}.issubset(click.columns)

    # 论文统计对照
    n_u = user["user_id"].nunique()
    n_s = streamer["streamer_id"].nunique()
    n_r = room["room_id"].nunique()
    print(f"  用户数={n_u} (论文23772)  主播数={n_s} (论文452621)  房间数={n_r} (论文11613708)")
    return user, streamer, room, click, neg


def build_item_meta(room, streamer):
    """构造 item_meta.csv: item_id(room_id) + i_ 前缀侧信息
    论文 CTR 有两种 item 定义：Item=Streamer / Item=Room
    这里默认 Item=Room（带 room 侧信息），同时 streamer 信息也并入"""
    print("[2/5] 构造 item_meta (room 级) ...")
    df = room.merge(streamer, on="streamer_id", how="left", suffixes=("", "_s"))
    # 离散特征 -> 整数编码 (ReChorus 要求 i_ 特征为整数)
    cat_cols = ["live_type", "live_content_category", "gender", "country",
                "device_brand", "live_operation_tag"]
    item_meta = pd.DataFrame()
    item_meta["item_id"] = df["room_id"]
    for c in cat_cols:
        if c in df.columns:
            codes = df[c].astype("category").cat.codes
            item_meta[f"i_{c}"] = codes + 1  # 0 留给 padding
    # dense 特征 (ReChorus 里非 _id/_c 结尾的走 Linear)
    for c in ["start_timestamp", "end_timestamp"]:
        if c in df.columns:
            item_meta[f"i_{c}"] = df[c].astype(np.int64)
    item_meta["i_streamer_id"] = df["streamer_id"].astype(np.int64)
    print(f"  item_meta shape: {item_meta.shape}, cols: {list(item_meta.columns)}")
    return item_meta


def build_user_meta(user):
    """构造 user_meta.csv: user_id + u_ 前缀侧信息"""
    print("[3/5] 构造 user_meta ...")
    cat_cols = ["gender", "country", "device_brand"]
    user_meta = pd.DataFrame()
    user_meta["user_id"] = user["user_id"]
    for c in cat_cols:
        if c in user.columns:
            codes = user[c].astype("category").cat.codes
            user_meta[f"u_{c}"] = codes + 1
    for c in ["age", "device_price", "fans_num", "follow_num",
              "accu_watch_live_cnt", "accu_watch_live_duration"]:
        if c in user.columns:
            user_meta[f"u_{c}"] = user[c].astype(np.float32)
    print(f"  user_meta shape: {user_meta.shape}")
    return user_meta


def leave_one_out_split(click, neg):
    """论文协议：leave-one-out
    对每个用户，按时间排序：
      最后 1 条正样本 -> test
      倒数第 2 条正样本 -> dev
      其余正样本 -> train
    负样本 (negative.csv) 全部归入 train (CTR 任务)
    """
    print("[4/5] leave-one-out 切分 ...")
    click_sorted = click.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    train_rows, dev_rows, test_rows = [], [], []
    for uid, grp in click_sorted.groupby("user_id"):
        idx = grp.index.values
        if len(idx) < 3:
            # 交互太少的用户全归 train
            train_rows.extend(idx.tolist())
            continue
        test_rows.append(idx[-1])
        dev_rows.append(idx[-2])
        train_rows.extend(idx[:-2].tolist())

    train_pos = click_sorted.loc[train_rows]
    dev_pos = click_sorted.loc[dev_rows]
    test_pos = click_sorted.loc[test_rows]
    print(f"  train_pos={len(train_pos)}  dev_pos={len(dev_pos)}  test_pos={len(test_pos)}")

    # 负样本：CTR 任务全部归入 train（论文用 negative.csv 做负样本）
    # dev/test 的负样本按论文 ReChorus preneg.py 逻辑：每正样本配 N 个随机负样本
    neg_train = neg.copy()

    def sample_negs(pos_df, n_neg=99):
        """为 dev/test 正样本采样负例 (随机未交互的 room)"""
        all_rooms = click["room_id"].unique()
        records = []
        pos_user_rooms = set(zip(click["user_id"], click["room_id"]))
        for _, row in pos_df.iterrows():
            uid = row["user_id"]
            seen = click_sorted[click_sorted["user_id"] == uid]["room_id"].unique()
            candidates = np.setdiff1d(all_rooms, seen)
            if len(candidates) == 0:
                continue
            n = min(n_neg, len(candidates))
            sampled = RNG.choice(candidates, size=n, replace=False)
            for r in sampled:
                records.append({"user_id": uid, "room_id": r,
                                "timestamp": row["timestamp"], "label": 0})
        return pd.DataFrame(records)

    dev_neg = sample_negs(dev_pos)
    test_neg = sample_negs(test_pos)
    print(f"  dev_neg={len(dev_neg)}  test_neg={len(test_neg)}  train_neg={len(neg_train)}")

    return train_pos, dev_pos, test_pos, neg_train, dev_neg, test_neg


def make_csv(pos, neg, label_pos=1):
    """合并正负样本，加 label 和 time 列"""
    pos = pos.copy()
    pos["label"] = label_pos
    pos["time"] = pos["timestamp"]
    cols = ["user_id", "room_id", "label", "time", "timestamp"]
    return pos[cols].rename(columns={"room_id": "item_id"})


def main():
    user, streamer, room, click, neg = load_raw()

    item_meta = build_item_meta(room, streamer)
    user_meta = build_user_meta(user)

    train_pos, dev_pos, test_pos, neg_train, dev_neg, test_neg = leave_one_out_split(click, neg)

    # 组装 train/dev/test
    neg_train = neg_train.copy()
    neg_train["label"] = 0
    neg_train["time"] = neg_train["timestamp"]

    def to_item_df(df):
        df = df.copy()
        if "time" not in df.columns:
            df["time"] = df["timestamp"]
        if "label" not in df.columns:
            df["label"] = 0
        return df[["user_id", "room_id", "label", "time", "timestamp"]].rename(columns={"room_id": "item_id"})

    train_df = pd.concat([make_csv(train_pos, None), to_item_df(neg_train)], ignore_index=True)
    dev_df = pd.concat([make_csv(dev_pos, None), to_item_df(dev_neg)], ignore_index=True)
    test_df = pd.concat([make_csv(test_pos, None), to_item_df(test_neg)], ignore_index=True)

    print(f"[5/5] 写出 ReChorus 兼容格式到 {OUT_DIR}")
    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    dev_df.to_csv(os.path.join(OUT_DIR, "dev.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)
    item_meta.to_csv(os.path.join(OUT_DIR, "item_meta.csv"), index=False)
    user_meta.to_csv(os.path.join(OUT_DIR, "user_meta.csv"), index=False)

    # 统计报告（与论文对照）
    stats = pd.DataFrame([{
        "n_users": user["user_id"].nunique(),
        "n_streamers": streamer["streamer_id"].nunique(),
        "n_rooms": room["room_id"].nunique(),
        "n_click_total": len(click),
        "n_negative_total": len(neg),
        "train_size": len(train_df),
        "dev_size": len(dev_df),
        "test_size": len(test_df),
        "train_pos_ratio": round(train_df["label"].mean(), 4),
        "test_pos_ratio": round(test_df["label"].mean(), 4),
    }])
    stats.to_csv(os.path.join(OUT_DIR, "data_stats.csv"), index=False)
    print("\n数据统计：")
    print(stats.to_string(index=False))
    print("\n预处理完成。文件清单：")
    for f in os.listdir(OUT_DIR):
        sz = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f"  {f}: {sz:.1f} KB")


if __name__ == "__main__":
    main()
