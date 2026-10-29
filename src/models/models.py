# ============================================================
# 模型定义
# - MLPEmbedding: 共享 embedding 底座
# - TimeDecay:    时间衰减编码（利用直播实时性）
# - DINAttention:  target-attention（简版 DIN）
# - MultiTaskMLP: shared-bottom 多任务（CTR / Like / Gift）
# ============================================================
import torch
import torch.nn as nn


class MLPEmbedding(nn.Module):
    """共享 embedding 底座：把稀疏 ID 查 embedding 后与 dense 特征拼接"""

    def __init__(self, vocab_sizes, dense_dim, emb_dim=32, hidden=(128, 64), dropout=0.2):
        super().__init__()
        # 每个稀疏列一个 embedding 表
        self.embeddings = nn.ModuleDict()
        for name, vsize in vocab_sizes.items():
            self.embeddings[name] = nn.Embedding(vsize, emb_dim, padding_idx=0 if name == "seq_sid" else None)
        total_emb = emb_dim * len(vocab_sizes)
        input_dim = total_emb + dense_dim

        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.out_dim = hidden[-1]

    def forward(self, sparse, dense, sparse_names):
        # sparse: (B, num_sparse_cols)
        embs = []
        for i, name in enumerate(sparse_names):
            embs.append(self.embeddings[name](sparse[:, i]))
        emb_cat = torch.cat(embs, dim=1)
        x = torch.cat([emb_cat, dense], dim=1)
        return self.trunk(x)


class TimeDecay(nn.Module):
    """会话时间衰减：对序列 embedding 按 exp(-dt/tau) 加权求和
    利用直播实时性：最近看的直播间权重更高"""

    def __init__(self, emb_dim=32, tau=24.0):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, seq_emb, seq_dt, mask=None):
        # seq_emb: (B, K, D)  seq_dt: (B, K) 小时
        # mask: (B, K) 1=有效 0=padding
        weight = torch.exp(-seq_dt / self.tau.clamp(min=1.0))
        if mask is not None:
            weight = weight * mask
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
        out = (seq_emb * weight.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return self.proj(out)


class DINAttention(nn.Module):
    """简版 DIN：target 对用户历史行为序列做注意力"""

    def __init__(self, emb_dim=32):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(emb_dim * 4, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, target_emb, seq_emb, mask=None):
        # target_emb: (B, D) 当前直播间对应主播 embedding
        # seq_emb: (B, K, D) 用户最近行为主播 embedding
        B, K, D = seq_emb.shape
        tgt = target_emb.unsqueeze(1).expand(-1, K, -1)  # (B,K,D)
        # 交互特征：差 / 积
        feat = torch.cat([tgt, seq_emb, tgt - seq_emb, tgt * seq_emb], dim=-1)
        score = self.attn(feat).squeeze(-1)  # (B,K)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        alpha = torch.softmax(score, dim=1)
        out = (alpha.unsqueeze(-1) * seq_emb).sum(dim=1)  # (B,D)
        return out


class WideDeep(nn.Module):
    """Wide & Deep: wide 线性部分 + deep MLP 部分"""

    def __init__(self, vocab_sizes, dense_dim, emb_dim=32, hidden=(128, 64), dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        for name, vsize in vocab_sizes.items():
            self.embeddings[name] = nn.Embedding(vsize, emb_dim)
        total_emb = emb_dim * len(vocab_sizes)

        # wide: 线性层直接吃 dense + 稀疏 embedding 的和
        self.wide = nn.Linear(total_emb + dense_dim, 1)

        # deep
        layers = []
        prev = total_emb + dense_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(prev, 1)

    def forward(self, sparse, dense, sparse_names):
        embs = [self.embeddings[name](sparse[:, i]) for i, name in enumerate(sparse_names)]
        emb_cat = torch.cat(embs, dim=1)
        x = torch.cat([emb_cat, dense], dim=1)
        wide_logit = self.wide(x)
        deep_logit = self.deep_out(self.deep(x))
        return wide_logit + deep_logit


class OursModel(nn.Module):
    """最终模型 = MLP 共享底座 + 多任务三头 + 时间衰减 + DIN 注意力
    创新点：
      1. shared-bottom 多任务（CTR 主任务 + Like/Gift 辅助任务）
      2. 会话时间衰减编码（直播实时性）
      3. 简版 DIN target-attention
    """

    def __init__(self, vocab_sizes, dense_dim, emb_dim=32, hidden=(128, 64), dropout=0.2,
                 use_timedecay=True, use_din=True, use_multitask=True):
        super().__init__()
        self.use_timedecay = use_timedecay
        self.use_din = use_din
        self.use_multitask = use_multitask

        # 主 embedding 底座
        self.embeddings = nn.ModuleDict()
        for name, vsize in vocab_sizes.items():
            self.embeddings[name] = nn.Embedding(vsize, emb_dim)
        total_emb = emb_dim * len(vocab_sizes)

        # DIN 序列 embedding（复用 streamer_id 的 embedding 表）
        self.seq_emb = self.embeddings["streamer_id"]

        # 附加模块输出维度
        extra = 0
        if use_timedecay:
            self.timedecay = TimeDecay(emb_dim)
            extra += emb_dim
        if use_din:
            self.din = DINAttention(emb_dim)
            extra += emb_dim

        trunk_in = total_emb + dense_dim + extra
        layers = []
        prev = trunk_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)

        # 任务头
        self.ctr_head = nn.Linear(prev, 1)
        if use_multitask:
            self.like_head = nn.Linear(prev, 1)
            self.gift_head = nn.Linear(prev, 1)

    def forward(self, sparse, dense, sparse_names, seq_sid=None, seq_dt=None):
        embs = [self.embeddings[name](sparse[:, i]) for i, name in enumerate(sparse_names)]
        emb_cat = torch.cat(embs, dim=1)
        parts = [emb_cat, dense]

        # target streamer embedding（DIN 用）
        # sparse_names 里 streamer_id 的索引
        sid_idx = sparse_names.index("streamer_id")
        target_emb = self.embeddings["streamer_id"](sparse[:, sid_idx])

        if self.use_timedecay or self.use_din:
            seq_emb = self.seq_emb(seq_sid)  # (B,K,D)
            mask = (seq_sid > 0).float()
            if self.use_timedecay:
                parts.append(self.timedecay(seq_emb, seq_dt, mask))
            if self.use_din:
                parts.append(self.din(target_emb, seq_emb, mask))

        x = torch.cat(parts, dim=1)
        h = self.trunk(x)
        ctr_logit = self.ctr_head(h).squeeze(-1)
        out = {"ctr": ctr_logit}
        if self.use_multitask:
            out["like"] = self.like_head(h).squeeze(-1)
            out["gift"] = self.gift_head(h).squeeze(-1)
        return out


# ============================================================
# MLP 优化模块（MLP++）
# 优化点：
#   1. 残差连接 + BatchNorm + GELU（稳定训练，可堆更深）
#   2. 显式特征交叉（user × streamer 逐元素积/差，类似 FM/DCN）
#   3. 行为序列池化（mean pooling 用户最近点击的主播 embedding）
#   4. 自适应 embedding 维度（ID 类特征用更大维度）
#   5. 多任务头（CTR / Like / Gift）
# ============================================================

class ResidualBlock(nn.Module):
    """残差 MLP 块：Linear → (BN) → Act → Dropout → Linear → (BN) + 残差"""

    def __init__(self, dim, dropout=0.1, use_bn=True, act="gelu"):
        super().__init__()
        act_cls = nn.GELU if act == "gelu" else nn.ReLU
        layers = [nn.Linear(dim, dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(dim))
        layers += [act_cls(), nn.Dropout(dropout), nn.Linear(dim, dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(dim))
        self.block = nn.Sequential(*layers)
        self.act = act_cls()

    def forward(self, x):
        return self.act(x + self.block(x))


class OptimizedMLP(nn.Module):
    """MLP++：优化版 CTR 预测模型
    相比基础 MLP 的改进：
      - 残差块 + (可选)BatchNorm + (可选)GELU（可训练更深，收敛更稳）
      - 显式 user×streamer 特征交叉（FM 式逐元素积 + 差，用全维 embedding）
      - 用户行为序列 mean-pooling（利用直播实时性）
      - 可选分层 embedding 维度（ID 特征更大维度）
      - 可选多任务（CTR + Like + Gift）
    经诊断调优后的最优默认配置：noBN + GELU + 2残差块 + uniform16 + dropout0.2 + multitask
    """

    def __init__(self, vocab_sizes, dense_dim, emb_dim=16, hidden=(64, 32),
                 dropout=0.2, use_cross=False, use_seq=False, use_multitask=True,
                 num_res_blocks=2, uniform_emb=True, use_bn=False, act="gelu"):
        super().__init__()
        self.use_cross = use_cross
        self.use_seq = use_seq
        self.use_multitask = use_multitask
        self.sparse_names = list(vocab_sizes.keys())
        act_cls = nn.GELU if act == "gelu" else nn.ReLU

        # embedding 维度：uniform_emb=True 时全部用 emb_dim（与 baseline 一致）；
        # 否则 ID 特征用更大维度、类别特征用更小维度
        self.emb_dims = {}
        for name in self.sparse_names:
            if uniform_emb:
                self.emb_dims[name] = emb_dim
            elif name in ("user_id", "streamer_id"):
                self.emb_dims[name] = max(emb_dim * 2, 32)
            else:
                self.emb_dims[name] = max(emb_dim // 2, 8)

        self.embeddings = nn.ModuleDict()
        for name, vsize in vocab_sizes.items():
            self.embeddings[name] = nn.Embedding(
                vsize, self.emb_dims[name],
                padding_idx=0 if name == "streamer_id" else None
            )

        total_emb = sum(self.emb_dims.values())

        # 特征交叉：user_id emb 与 streamer_id emb 逐元素交互
        cross_dim = 0
        if use_cross:
            cross_dim = self.emb_dims["user_id"] * 2  # product + difference

        # 序列池化维度
        seq_dim = 0
        if use_seq:
            self.seq_proj = nn.Linear(self.emb_dims["streamer_id"], 16)
            seq_dim = 16

        input_dim = total_emb + dense_dim + cross_dim + seq_dim

        # 第一层投影到 hidden[0]
        layers = [nn.Linear(input_dim, hidden[0])]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden[0]))
        layers += [act_cls(), nn.Dropout(dropout)]
        prev = hidden[0]

        # 残差块
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(prev, dropout, use_bn=use_bn, act=act))

        # 后续降维层
        for h in hidden[1:]:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers += [act_cls(), nn.Dropout(dropout)]
            prev = h

        self.trunk = nn.Sequential(*layers)
        self.out_dim = prev

        # 任务头
        self.ctr_head = nn.Linear(prev, 1)
        if use_multitask:
            self.like_head = nn.Linear(prev, 1)
            self.gift_head = nn.Linear(prev, 1)

    def forward(self, sparse, dense, sparse_names, seq_sid=None, seq_dt=None):
        embs = []
        for i, name in enumerate(sparse_names):
            embs.append(self.embeddings[name](sparse[:, i]))
        emb_cat = torch.cat(embs, dim=1)
        parts = [emb_cat, dense]

        # 显式特征交叉：user × streamer
        if self.use_cross:
            uid_idx = sparse_names.index("user_id")
            sid_idx = sparse_names.index("streamer_id")
            u_emb = self.embeddings["user_id"](sparse[:, uid_idx])
            s_emb = self.embeddings["streamer_id"](sparse[:, sid_idx])
            cross_prod = u_emb * s_emb       # 逐元素积（FM 思想）
            cross_diff = u_emb - s_emb       # 差（捕捉偏好方向）
            parts.append(torch.cat([cross_prod, cross_diff], dim=1))

        # 行为序列 mean-pooling
        if self.use_seq and seq_sid is not None:
            seq_emb = self.embeddings["streamer_id"](seq_sid)  # (B,K,D)
            mask = (seq_sid > 0).float().unsqueeze(-1)  # (B,K,1)
            seq_sum = (seq_emb * mask).sum(dim=1)
            seq_cnt = mask.sum(dim=1).clamp(min=1.0)
            seq_mean = seq_sum / seq_cnt  # (B,D)
            parts.append(self.seq_proj(seq_mean))

        x = torch.cat(parts, dim=1)
        h = self.trunk(x)
        out = {"ctr": self.ctr_head(h).squeeze(-1)}
        if self.use_multitask:
            out["like"] = self.like_head(h).squeeze(-1)
            out["gift"] = self.gift_head(h).squeeze(-1)
        return out
