# ============================================================
# 优化模型 —— FI-DeepFM (Feature-Interaction enhanced DeepFM)
# 针对 DeepFM 的两个具体弱点设计（非凭空缝合）：
#   弱点1: DeepFM 对所有特征 embedding 平等对待，无重要性区分
#          → 加 FeatureGating: 对每个特征 embedding 做门控加权
#   弱点2: DeepFM 只靠 MLP 隐式建模高阶交叉，FM 只到二阶
#          → 加 InteractingLayer (AutoInt, CIKM'19): 显式多头自注意力建模高阶交叉
# 同口径：共享 DeepFM 的 embedding、FM 部分、损失、评估，只替换 deep 部分
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from reproduce.models_repro import DeepFMModel


class FeatureGating(nn.Module):
    """特征门控：为每个特征的 embedding 生成一个标量权重，动态抑制噪声特征"""

    def __init__(self, num_fields, emb_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(num_fields * emb_dim, num_fields),
            nn.Sigmoid(),
        )

    def forward(self, fm_vectors):
        # fm_vectors: (B, F, D)
        B, F, D = fm_vectors.shape
        weights = self.gate(fm_vectors.reshape(B, -1))   # (B, F)
        return fm_vectors * weights.unsqueeze(-1)        # (B, F, D)


class InteractingLayer(nn.Module):
    """AutoInt 交互层 (CIKM'19): 特征间多头自注意力，显式建模高阶交叉"""

    def __init__(self, emb_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        # x: (B, F, D)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm(x + self.ffn(x))
        return x


class FIDeepFMModel(DeepFMModel):
    """FI-DeepFM: 在 DeepFM 基础上加 FeatureGating + AutoInt 交互层
    与 DeepFM 同口径：共享 embedding/FM/损失/评估，deep 部分增强。
    """

    def __init__(self, feature_max, context_features, emb_size=64, layers=(64,),
                 dropout=0.1, num_heads=2, num_interact_layers=1):
        super().__init__(feature_max, context_features, emb_size, layers, dropout)
        self.num_fields = len(context_features)
        self.gating = FeatureGating(self.num_fields, emb_size)
        self.interact_layers = nn.ModuleList([
            InteractingLayer(emb_size, num_heads, dropout) for _ in range(num_interact_layers)
        ])
        # deep 输入：门控+交互后的特征 flatten，与原 DeepFM 同维度
        pre_size = self.num_fields * emb_size
        deep_layers = []
        prev = pre_size
        for h in layers:
            deep_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        deep_layers.append(nn.Linear(prev, 1))
        self.deep_layers = nn.Sequential(*deep_layers)
        for m in self.deep_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feed_dict):
        fm_vectors, linear_value = self.fm._get_embeddings(feed_dict)
        # FM 部分（与 DeepFM 完全一致）
        fm = 0.5 * (fm_vectors.sum(dim=1).pow(2) - fm_vectors.pow(2).sum(dim=1)).sum(dim=-1)
        fm_pred = fm + linear_value
        # 优化部分：门控 → 自注意力交互 → flatten → MLP
        x = self.gating(fm_vectors)
        for layer in self.interact_layers:
            x = layer(x)
        deep_pred = self.deep_layers(x.flatten(start_dim=1)).squeeze(-1)
        return {"logit": fm_pred + deep_pred}
