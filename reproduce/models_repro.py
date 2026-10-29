# ============================================================
# Baseline 模型 —— 严格对齐 ReChorus 实现
# 参考: THUwangcy/ReChorus src/models/context/{FM,WideDeep,DeepFM}.py
# 关键对齐点：
#   1. FM: linear_embedding (1维) + context_embedding (emb_size维) + 二阶交叉
#   2. WideDeep: FM wide 部分 + MLP deep 部分，embedding 共享
#   3. DeepFM: FM 二阶交叉项 + MLP，共享 embedding，无 wide 之外的额外输入
#   4. 特征处理：_id/_c 结尾走 Embedding，其余走 Linear(1->dim)
#   5. 默认超参: emb_size=64, layers=[64], dropout=0, lr=1e-3 (ReChorus 默认)
# ============================================================
import torch
import torch.nn as nn
import numpy as np


class FMModel(nn.Module):
    """Factorization Machine (对齐 ReChorus FMBase)
    prediction = overall_bias + sum(linear) + 0.5*sum((sum emb)^2 - sum(emb^2))
    """

    def __init__(self, feature_max, context_features, emb_size=64):
        super().__init__()
        self.vec_size = emb_size
        self.context_features = context_features
        self.context_embedding = nn.ModuleDict()
        self.linear_embedding = nn.ModuleDict()
        for f in context_features:
            if f.endswith("_c") or f.endswith("_id"):
                self.context_embedding[f] = nn.Embedding(feature_max[f], emb_size)
                self.linear_embedding[f] = nn.Embedding(feature_max[f], 1)
            else:
                self.context_embedding[f] = nn.Linear(1, emb_size, bias=False)
                self.linear_embedding[f] = nn.Linear(1, 1, bias=False)
        self.overall_bias = nn.Parameter(torch.tensor([0.01]))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_embeddings(self, feed_dict):
        fm_vectors, linear_values = [], []
        for f in self.context_features:
            x = feed_dict[f]
            if f.endswith("_c") or f.endswith("_id"):
                v = self.context_embedding[f](x)        # (B, D)
                lv = self.linear_embedding[f](x)         # (B, 1)
            else:
                v = self.context_embedding[f](x.float().unsqueeze(-1))
                lv = self.linear_embedding[f](x.float().unsqueeze(-1))
            fm_vectors.append(v)
            linear_values.append(lv)
        fm_vectors = torch.stack(fm_vectors, dim=1)      # (B, F, D)
        linear_value = self.overall_bias + torch.cat(linear_values, dim=-1).sum(dim=-1)
        return fm_vectors, linear_value

    def forward(self, feed_dict):
        fm_vectors, linear_value = self._get_embeddings(feed_dict)
        # 二阶交叉: 0.5 * (sum^2 - sum_of_squares)
        fm = 0.5 * (fm_vectors.sum(dim=1).pow(2) - fm_vectors.pow(2).sum(dim=1)).sum(dim=-1)
        return {"logit": (linear_value + fm)}


class WideDeepModel(nn.Module):
    """Wide & Deep (对齐 ReChorus WideDeepBase)
    wide = FM linear 部分; deep = MLP(flatten embeddings)
    """

    def __init__(self, feature_max, context_features, emb_size=64, layers=(64,), dropout=0.0):
        super().__init__()
        self.fm = FMModel(feature_max, context_features, emb_size)
        self.vec_size = emb_size
        self.context_features = context_features
        pre_size = len(context_features) * emb_size
        deep_layers = []
        prev = pre_size
        for h in layers:
            deep_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        deep_layers.append(nn.Linear(prev, 1))
        self.deep_layers = nn.Sequential(*deep_layers)
        self.fm._init_weights()
        for m in self.deep_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feed_dict):
        fm_vectors, wide_pred = self.fm._get_embeddings(feed_dict)
        deep_vec = fm_vectors.flatten(start_dim=1)
        deep_pred = self.deep_layers(deep_vec).squeeze(-1)
        return {"logit": deep_pred + wide_pred}


class DeepFMModel(nn.Module):
    """DeepFM (对齐 ReChorus DeepFMBase)
    FM 二阶交叉 + MLP，共享 embedding。比 WideDeep 多了 FM 二阶交叉项。
    """

    def __init__(self, feature_max, context_features, emb_size=64, layers=(64,), dropout=0.0):
        super().__init__()
        self.fm = FMModel(feature_max, context_features, emb_size)
        self.vec_size = emb_size
        self.context_features = context_features
        pre_size = len(context_features) * emb_size
        deep_layers = []
        prev = pre_size
        for h in layers:
            deep_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        deep_layers.append(nn.Linear(prev, 1))
        self.deep_layers = nn.Sequential(*deep_layers)
        self.fm._init_weights()
        for m in self.deep_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feed_dict):
        fm_vectors, linear_value = self.fm._get_embeddings(feed_dict)
        # FM 部分
        fm = 0.5 * (fm_vectors.sum(dim=1).pow(2) - fm_vectors.pow(2).sum(dim=1)).sum(dim=-1)
        fm_pred = fm + linear_value
        # Deep 部分
        deep_pred = self.deep_layers(fm_vectors.flatten(start_dim=1)).squeeze(-1)
        return {"logit": fm_pred + deep_pred}
