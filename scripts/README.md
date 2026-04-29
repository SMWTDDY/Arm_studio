# scripts

项目级脚本入口目录。

保留边界：

```text
data_tools/            数据转换、数据验证、训练缓存、视频导出
diagnostics/           仿真环境、渲染、相机和视觉诊断
model_tools/           checkpoint 检查
agent_factory/         agent_factory 相关 H5 检查、转换和 smoke trial
piper/                 Piper CAN、Orbbec、统一采集、真实/仿真采集与回放
realman/               历史 Realman 环境的采集、合并、回放和 smoke test
sim/                   旧 ArmStudio 仿真采集入口
training/              原 Piper_test/script 中仍有价值的训练与验证脚本
```

已删除旧兼容壳和断掉的入口：`record.sh`、`inference.sh`、`legacy/`、
`agent_infra/Piper_Env/Script/`、`agent_infra/Realman_Env/Script/`、
`agent_factory/script/`、`piper_scripts/`、旧副本目录 `test piper/`。

本目录只保留可直接运行或仍有维护价值的脚本；断掉的固定路径壳脚本
`scripts/agent_factory/convert.sh` 已移除，数据转换直接调用
`scripts/agent_factory/convert_to_flattened.py`。

目录约定：

```text
datasets/              H5、LeRobot、训练缓存、在线采样 replay 数据
outputs/               checkpoint、推理结果、debug 图、导出视频
```
