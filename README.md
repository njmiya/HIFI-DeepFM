# HIFI-DeepFM: Mitigating Representation Failure and Exposure Bias in Live Streaming Recommendation

This repository is the official implementation of the paper **“HIFI-DeepFM: Mitigating Representation Failure and Exposure Bias in Live Streaming Recommendation via History-Interaction Feature Integration”**.

**HIFI-DeepFM** is a lightweight network that combines explicit feature crosses with time-aware statistical history features. It acts as a **feature-side fallback** for failed identity representations under passive exposure and extreme cold start, without introducing heavy computational costs.

> 📄 **Paper Status**: Manuscript prepared for *Decision Support Systems* / *Information & Management*.

---

## 📖 Overview

Live-streaming platforms present rooms in an auto-playing feed where feedback is extremely sparse. Roughly **70% of streamers receive fewer than five interactions** over a 21-day window on KuaiLive, making ID embeddings unreliable. HIFI-DeepFM addresses this by aggregating **11 pre-exposure statistics** across 4 granularities (user, room, user×streamer, streamer)—including exposure counts, realized CTRs, and recency intervals—and injecting them via a **zero-initialized linear residual** that can be learned end-to-end.

### ✅ Advantages

*   **Lightweight fallback**: Only adds **132K parameters (+0.14%)** and a single CPU-process latency increase from 0.62 ms to 1.89 ms.
*   **Zero negative transfer**: On dense datasets like KuaiRec, the full model’s weakest seed (0.8349) still exceeds the DeepFM baseline’s strongest seed (0.8322).
*   **Targeted rescue**: Cold-start streamers achieve a massive **+22.62% relative AUC gain**, while head entities remain neutral (near-ceiling 0.9524 AUC shows no measurable drop).
*   **Strict temporal integrity**: All 11 statistics are computed using **only events timestamped strictly before the prediction time**, preventing future leakage.
*   **Evidence of representation fallback**: In an **unseen-user experiment**, history features lift AUC from 0.8903 to 0.9381 when user-ID embeddings are completely untrained.

---

## 📊 Datasets

We use two public datasets from the Kuaishou ecosystem:

| Dataset | Positive Rate | Exposure Mode | Cold-Start Severity | History Feature Gain |
| :--- | :--- | :--- | :--- | :--- |
| **KuaiLive** | ≈ 19.6% | Passive (platform feed) | Severe (70% streamers < 5 interactions) | **+3.59%** |
| **KuaiRec** | 16.4% | Passive (auto-play) | Minimal (99.87% test videos warm) | **+0.68%** |

*   **KuaiLive**: [Download Link](https://imgkkk574.github.io/KuaiLive/)
*   **KuaiRec**: [Download Link](https://kuairec.com/)

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n hifi python=3.8
conda activate hifi
pip install -r requirements.txt