#!/bin/bash

# 默认运行真机离线训练
# python3 scripts/training/train_realman_offline.py
python3 scripts/training/train_realman_critic_only.py --config outputs/checkpoints/0325/config.yaml
