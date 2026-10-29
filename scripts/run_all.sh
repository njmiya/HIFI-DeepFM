#!/bin/bash
# 一键运行全流程
cd "$(dirname "$0")/.."
python3 src/data/synthetic_data.py
python3 src/data/feature_eng.py
python3 main.py
