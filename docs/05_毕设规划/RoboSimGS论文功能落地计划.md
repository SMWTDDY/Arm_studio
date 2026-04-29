# RoboSimGS 论文功能落地计划

本文档根据论文 `High-Fidelity Simulated Data Generation for Real-World Zero-Shot Robotic Manipulation Learning with Gaussian Splatting` 和本地 `3DGS` 库整理，目标是把论文中的关键能力拆成 `Arm_studio` 可实现、可验收、可写进毕设的工程模块。

参考材料：

```text
docs/06_参考材料/High-Fidelity_Simulated_Data_Generation_for_RoboSimGS.pdf
/home/wry/Desktop/NewArm_studio/3DGS/
```

## 1. 论文功能拆解

论文主线可以压缩成一句话：

```text
静态真实场景用 3DGS 保真渲染，交互物体用 mesh/URDF 保证物理，再通过 MLLM 估计物理和关节结构，最后把真实场景、仿真世界和真实相机对齐。
```

你要求实现的四个功能对应如下：

```text
外部模型导入
  -> 把 GLB/OBJ/URDF/mesh 资产导入 SAPIEN/ManiSkill，成为可碰撞、可控制、可随机化的仿真物体。

静态 3D 高斯背景生成
  -> 从真实桌面多视角图像生成 Gaussian PLY，并作为仿真视觉背景，而不是物理碰撞对象。

物理属性和运动学结构 AI 自动估计
  -> 从多视角图像或 mesh 渲染图估计 density、friction、mass、joint type、axis、limit，并输出 physics.json / articulation.json / URDF。

虚拟环境和现实相机校准
  -> 世界坐标用机器人几何点云 ICP 对齐；相机位姿用真实图像和 3DGS 渲染图的光度误差优化或半自动搜索对齐。
```

## 2. 本地 3DGS 代码可复用清单

### 2.1 外部模型与 URDF 生成

相关文件：

```text
3DGS/Articulation/articulation_inference.py
3DGS/Articulation/urdf_generation/pipeline.py
3DGS/Articulation/urdf_generation/hinge_detector.py
3DGS/Articulation/urdf_generation/urdf_builder.py
3DGS/Articulation/utils/mesh_utils.py
```

已有能力：

```text
读取 GLB
交互式分割 movable/static parts
根据两个 mesh 的接触区域估计 hinge position 和 hinge axis
生成 revolute joint URDF
保存 metadata.json
```

主要缺口：

```text
当前更像 openbox/hinge demo，默认只覆盖 revolute joint。
还没有统一资产目录规范。
还没有 SAPIEN/ManiSkill asset loader。
还没有对 prismatic drawer、普通刚体 OBJ、已有 URDF 的统一导入接口。
```

### 2.2 静态 3DGS 背景

相关文件：

```text
3DGS/README.md
3DGS/Gaussians/render.py
3DGS/Gaussians/renderer_ogl.py
3DGS/Gaussians/util_gau.py
```

已有能力：

```text
README 给出使用 Nerfstudio / Feature Splatting 重建背景的路线。
render.py 可以加载 Gaussian PLY 并用 CUDA rasterizer 渲染。
```

主要缺口：

```text
背景训练脚本不在本库内，需要外接 Nerfstudio 或 graphdeco/gsplat。
render.py 存在硬编码路径和相机参数。
当前 renderer 没有 Arm_studio 风格配置、没有和 SAPIEN 相机参数打通。
```

### 2.3 AI 估计物理属性和运动学结构

相关文件：

```text
3DGS/Articulation/physics_estimation.py
3DGS/Articulation/material/material_table.json
3DGS/Articulation/material/material_table_pbd.json
3DGS/Articulation/urdf_generation/pipeline.py
3DGS/Articulation/utils/gpt_utils.py
```

已有能力：

```text
通过 BLIP2/CLIP/GPT 风格流程估计材料类别和物理参数。
URDF pipeline 可用 GPT 推荐 joint limit、friction、damping、mass 等参数。
```

主要缺口：

```text
physics_estimation.py 中材料表路径和当前目录结构不完全一致。
依赖大模型、OpenAI key、本地 BLIP2/CLIP 权重，直接跑通风险高。
输出不是 Arm_studio 统一 physics.json。
需要人工确认机制，不能把 AI 输出直接当真值。
```

### 2.4 虚拟环境与现实相机校准

相关文件：

```text
3DGS/utils/icp.py
3DGS/utils/mesh2pc.py
3DGS/Gaussians/render.py
```

已有能力：

```text
icp.py 可以对真实重建点云和仿真机器人点云做 ICP 对齐。
mesh2pc.py 可辅助从 mesh 采样点云。
render.py 可作为可微/近似可微渲染的起点。
```

主要缺口：

```text
论文中的 Camera Pose Alignment 代码没有公开。
icp.py 的命令行参数说明和代码实现不一致，size 需要转 float。
icp.py 目前不保存 world_alignment.json。
相机位姿优化需要自己实现粗标定 + 局部搜索/梯度优化。
```

## 3. 建议新增工程结构

建议在 `Arm_studio` 内新增以下目录，避免直接修改 `3DGS` 原库：

```text
assets/realsim/
  desktop_sorting/
    background/
      gaussian.ply
      train_notes.md
    objects/
      block_a/
        mesh.obj
        collision.obj
        physics.json
      drawer/
        source.glb
        articulated.urdf
        articulation.json
    cameras.json
    world_alignment.json
    scene_config.yaml

sim_assets/
  importers/
    mesh_importer.py
    urdf_importer.py
    sapien_asset_loader.py
  schemas.py

gs_bridge/
  gaussian_renderer.py
  camera_calibration.py
  world_alignment.py
  composite_renderer.py

scripts/realsim/
  import_object_asset.py
  build_background_3dgs.md
  estimate_asset_physics.py
  estimate_asset_articulation.py
  align_world_frame.py
  calibrate_camera_pose.py
  preview_gs_sapien_scene.py
```

说明：

```text
assets/realsim/ 保存具体场景资产。
sim_assets/ 负责把外部模型变成 SAPIEN/ManiSkill 可加载资产。
gs_bridge/ 负责 3DGS 渲染、世界对齐、相机标定和图像合成。
scripts/realsim/ 放用户可运行脚本。
```

## 4. 功能一：外部模型导入

目标：

```text
支持 OBJ/GLB/URDF 三类外部模型导入，并能在 SAPIEN/ManiSkill 中作为物理对象出现。
```

输入：

```text
mesh.obj / mesh.glb / object.urdf
scale
mass
friction
collision_mode
initial_pose
```

输出：

```text
assets/realsim/<scene>/objects/<name>/
  visual.*
  collision.*
  physics.json
  articulation.json 可选
  object_config.yaml
```

最小实现：

1. `import_object_asset.py` 读取外部 OBJ/GLB，生成统一目录。
2. 用 `trimesh` 做尺度归一、坐标轴检查、碰撞 mesh 简化。
3. 刚体物体先生成 `physics.json`，字段包含 `mass`、`density`、`friction`、`restitution`。
4. 已有 URDF 直接复制并记录 mesh 相对路径。
5. 在 ManiSkill/SAPIEN 场景中通过 config 加载这些资产。

验收标准：

```text
能导入一个 block/cube/banana 类刚体 mesh。
能导入一个 drawer/box 类 URDF。
仿真中能看到、能碰撞、能被随机化初始位姿。
资产目录中能追溯 scale、mass、friction 和来源文件。
```

## 5. 功能二：静态 3DGS 背景生成

目标：

```text
从真实桌面场景生成 Gaussian PLY，并在仿真渲染时作为背景。
```

输入：

```text
多视角 RGB 图片
相机内参或 COLMAP 估计结果
可选语义 mask：robot/background/object
```

输出：

```text
assets/realsim/<scene>/background/gaussian.ply
assets/realsim/<scene>/background/transforms.json
assets/realsim/<scene>/background/reconstruction_report.md
```

最小实现：

1. 先采用外部工具生成 Gaussian PLY：Nerfstudio、graphdeco 或 gsplat 三选一。
2. 将生成结果复制到 `assets/realsim/<scene>/background/`。
3. 重构 `3DGS/Gaussians/render.py` 思路，写成 `gs_bridge/gaussian_renderer.py`。
4. renderer 不再写死路径，相机参数从 `cameras.json` 或 SAPIEN camera 读取。
5. 第一版只要求离线渲染指定相机视角，第二版再接入仿真 step。

验收标准：

```text
给定 camera pose，能渲染出桌面背景图。
相机分辨率、内参、外参都来自配置。
渲染结果可保存到 outputs/realsim_preview/。
论文中“3DGS 负责静态背景，mesh 负责交互物体”的边界在代码里保持清楚。
```

## 6. 功能三：AI 自动估计物理属性

目标：

```text
从物体图像或 mesh 渲染视图估计仿真需要的物理参数，但必须保留人工审核入口。
```

输入：

```text
物体四视角图片或 mesh 渲染图
物体名称
材质候选
尺寸估计
```

输出：

```text
physics.json
```

建议 schema：

```json
{
  "object_name": "drawer",
  "source": "ai_estimated_with_human_review",
  "mass_kg": 0.5,
  "density_kg_m3": 700,
  "friction_static": 0.6,
  "friction_dynamic": 0.4,
  "restitution": 0.05,
  "youngs_modulus_pa": null,
  "poisson_ratio": null,
  "confidence": 0.6,
  "review_status": "pending"
}
```

最小实现：

1. 先把 `3DGS/Articulation/material/*.json` 复制或引用为材料先验表。
2. 写 `estimate_asset_physics.py`，支持两种模式：
   `--mode manual-template` 生成可编辑模板；
   `--mode ai-suggest` 调用 MLLM 生成建议。
3. AI 输出必须落到 `physics.suggested.json`，人工确认后才生成 `physics.json`。
4. 训练/仿真只读取 `physics.json`，不直接读取未审核建议。

验收标准：

```text
无 API key 时也能生成 manual physics.json 模板。
有 API key 时能生成 suggested 参数和 reasoning。
每个参数都有单位和来源。
SAPIEN/ManiSkill 加载时能把 friction/mass 应用到物体。
```

## 7. 功能四：AI 自动估计运动学结构

目标：

```text
从静态 mesh 推断刚体/铰链/滑轨结构，生成 articulation.json 和 URDF。
```

输入：

```text
source.glb
多视角渲染图
部件语义标签，例如 body、lid、drawer
可选人工点选分割结果
```

输出：

```text
articulation.json
articulated.urdf
metadata.json
```

建议 schema：

```json
{
  "object_name": "drawer",
  "joint_type": "prismatic",
  "parent_link": "body",
  "child_link": "drawer",
  "axis": [1, 0, 0],
  "origin_xyz": [0, 0, 0],
  "limit_lower": 0.0,
  "limit_upper": 0.18,
  "damping": 0.2,
  "friction": 0.5,
  "confidence": 0.55,
  "review_status": "pending"
}
```

最小实现：

1. 复用 `3DGS/Articulation/articulation_inference.py` 的交互式分割思路。
2. 第一版只支持：
   刚体 `fixed`；
   盒盖/门 `revolute`；
   抽屉 `prismatic`。
3. `revolute` 可复用 `HingeDetector` 的接触区域 PCA 估计。
4. `prismatic` 第一版允许 AI 给建议、人工确认 axis 和 limit。
5. 所有 AI 估计必须生成 `articulation.suggested.json`，确认后生成 `articulation.json` 和 URDF。

验收标准：

```text
一个盒盖/门类物体能生成 revolute URDF。
一个抽屉类物体能生成 prismatic URDF 或人工确认后的 articulation.json。
仿真中能通过 joint qpos 控制开合/滑动。
失败时能退化为刚体导入，不阻塞主任务。
```

## 8. 功能五：虚拟环境和现实相机校准

论文中校准包含两层：世界坐标对齐和相机位姿对齐。建议分开实现。

### 8.1 世界坐标对齐

目标：

```text
把 3DGS 重建坐标系对齐到 SAPIEN/ManiSkill 世界坐标系。
```

输入：

```text
真实重建中的机器人点云 real_robot.ply
仿真 URDF 采样出的机器人点云 sim_robot.ply
人工 scale 初值
```

输出：

```text
world_alignment.json
```

最小实现：

1. 修复并迁移 `3DGS/utils/icp.py` 到 `gs_bridge/world_alignment.py`。
2. 支持 `--real-pcd`、`--sim-pcd`、`--scale`、`--output` 参数。
3. `scale` 转为 float。
4. 输出 4x4 transform、mean distance、iterations、scale。
5. 提供可视化预览，但保存 JSON 不依赖 GUI。

验收标准：

```text
world_alignment.json 可被后续 renderer 和 asset loader 读取。
对齐误差能记录在 report 中。
同一 scene 下所有 object/camera/gs 都使用同一个 world transform。
```

### 8.2 相机位姿对齐

目标：

```text
求出仿真相机 pose，使 3DGS 渲染图尽量匹配真实相机图片。
```

输入：

```text
real_reference.png
gaussian.ply
cameras.json 初始内参/外参
world_alignment.json
```

输出：

```text
cameras.json 更新后的 extrinsics
camera_alignment_report.md
preview_before.png
preview_after.png
```

最小实现路线：

1. 第一版：人工粗调 + 网格搜索。
   在初始 pose 附近随机/网格扰动 xyz 和 rpy，用 L1/SSIM 选择误差最小者。
2. 第二版：局部优化。
   如果 renderer 支持梯度，就优化 SE(3) 参数；否则使用 scipy/CMA-ES/Nelder-Mead 这类无梯度优化。
3. 第三版：加入 mask。
   只对静态背景区域计算 photometric loss，避开真实机器人和可移动物体干扰。

验收标准：

```text
能输出 before/after 对比图。
相机 pose 更新写回 cameras.json。
同一个 camera pose 可用于 GS 渲染和 SAPIEN 前景渲染。
文档中明确说明：论文原版 Camera Pose Alignment 未公开，本项目实现的是可复现近似版本。
```

## 9. 推荐实现顺序

第一阶段：资产和刚体闭环。

```text
1. 建立 assets/realsim/desktop_sorting/ 目录规范。
2. 实现 import_object_asset.py。
3. 实现 physics.json manual-template。
4. 在 SAPIEN/ManiSkill 中加载一个外部刚体 mesh。
```

第二阶段：3DGS 背景闭环。

```text
1. 用外部工具生成桌面 gaussian.ply。
2. 实现 gaussian_renderer.py，去掉 3DGS/render.py 硬编码路径。
3. 用 cameras.json 渲染指定视角。
4. 输出 preview。
```

第三阶段：对齐闭环。

```text
1. 修复并迁移 ICP，输出 world_alignment.json。
2. 实现相机粗调/网格搜索，输出 before/after。
3. 让 GS 背景与仿真相机共用同一套 camera config。
```

第四阶段：AI 估计闭环。

```text
1. 物理参数 AI suggested -> 人工确认 -> physics.json。
2. revolute 物体：交互分割 -> HingeDetector -> URDF。
3. prismatic 物体：AI/人工 axis+limit -> URDF。
4. SAPIEN/ManiSkill 加载并控制 articulation。
```

第五阶段：论文实验闭环。

```text
1. 生成 sim-only 数据。
2. 生成 GS+sim 数据。
3. 与少量 real 数据合并训练。
4. 对比 real-only、sim-only、GS+sim、GS+sim+few-real。
```

## 10. 毕设最低完成标准

为了保证能完成答辩，不要把四个功能都做成“全自动研究级”。最低标准建议是：

```text
外部模型导入：支持刚体 mesh + 至少一个 URDF articulated object。
静态 3DGS 背景：能渲染真实桌面背景，并与仿真相机视角大致一致。
物理属性 AI 估计：能生成 suggested physics.json，并有人审确认流程。
运动学结构 AI 估计：能对盒盖/抽屉类物体生成或辅助生成 articulation.json/URDF。
虚实相机校准：有 world_alignment.json + cameras.json + before/after 预览图。
```

这些结果足够支撑论文中的“复现并工程化嵌入”叙事，也不会被相机优化、全自动分割或复杂关节推断拖死。

## 11. 论文写法建议

论文中建议把这部分写成：

```text
基于 RoboSimGS 思想的真实场景可交互仿真资产构建模块
```

创新点不要写成“完全复现 RoboSimGS”，而应写成：

```text
将 3DGS 静态背景、mesh/URDF 交互物体、AI 物理/运动学估计、虚实坐标/相机对齐，改造成适配低成本 Piper 机械臂和 Arm_studio 统一数据协议的工程化流程。
```

这样既承认论文方法来源，也能突出你自己的系统整合和毕设工程贡献。
