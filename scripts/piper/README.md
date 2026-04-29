# scripts/piper

Piper 统一环境强相关脚本统一放在这里。旧的
`agent_infra/Piper_Env/Script/` 已移除，避免出现两个脚本根目录。

目录边界：

```text
collect_unified.py     实验性统一配置 dry-run，不作为日常采集入口
test_unified_env.py    统一 real/sim 配置与协议 dry-run/smoke test
can/                   Piper USB-CAN 激活和枚举
camera/                Orbbec 标定、RGB/depth 检查、SDK 探针
sim/                   Piper 仿真 H5/LeRobot 数据采集脚本
single/                单臂采集、回放、H5/LeRobot 转换兼容脚本
dual/                  双臂采集、回放、H5/LeRobot 转换兼容脚本
```

默认数据根目录为 `datasets/piper`。真实和仿真采集都应写入这里；训练、
推理、调试图片和导出视频写入 `outputs/`。

仿真采集示例：

```bash
bash scripts/piper/sim/collect_h5.sh --arm-side right --mode joint
bash scripts/piper/sim/collect_h5.sh --arm-side left --mode joint
bash scripts/piper/sim/collect_lerobot.sh --arm-side right --mode joint
```

`--arm-side right` 加载 `piper_arm_right` 和右腕相机 URDF，默认读取 `can_mr`；
`--arm-side left` 加载 `piper_arm_left` 和左腕相机 URDF，默认读取 `can_ml`。
旧的 `--arm-mode right_hand/left_hand` 会被兼容映射到同一逻辑。

`sim/collect_lerobot.sh` 沿用 `scripts/sim/collect_data.py` 的 RealToSimTeleop
采集方式，退出后只把本次新增的 H5 自动转换成 LeRobot，避免重复转换历史轨迹。

原 Piper_test/trial 临时实验脚本不再保留在仓库中；Piper_test 已由用户外部备份。
