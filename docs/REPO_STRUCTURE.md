# Arm Studio Repository Structure

This repository keeps runtime code, data collection, training, diagnostics, and generated artifacts in separate areas. The top-level scripts that existed before the cleanup are still present as compatibility entry points.

## Core Runtime

```text
robot/
  base_arm.py
  piper/
    agent.py
    pose_ik.py
    piper_assets/

environments/
  conveyor_env.py
  grasping_env.py

teleop/
  get_pose.py
  keyboard_ik.py
  real_to_sim.py
```

`robot/`, `environments/`, and `teleop/` are the core simulation/control packages. Keep imports stable here because data collection, validation, and inference depend on these paths.

## Data Collection And Datasets

```text
data/
  recorder.py
  auto_getting_data.py
  check_data.py
  dataset_viewer.py

datasets/
  piper_joint_recording_*.hdf5
  piper_pose_recording_*.hdf5
  auto_collected_real/
  unified_pose/              # recommended output for mixed joint/pose conversion
```

Raw and generated HDF5 files are treated as local artifacts. For pose training, write standardized files to `datasets/unified_pose`.

Auto-collection design notes live in `docs/auto_collection_pseudocode.txt`.
Training and data usage notes live in `docs/TRAINING_AND_DATA_USAGE.md` and `docs/USAGE.md`.

## Training And Inference

```text
models/DiffusionPolicy/
  action_codec.py
  model.py
  policy.py
  vision_encoder.py

training/Diffusion_Training/
  training_config.py
  train_diffusion_vision.py

inference/
  client.py
  server.py
  create_dummy_model.py
```

Training checkpoints and loss plots now go under `outputs/checkpoints/vision`.

## Scripts

```text
scripts/
  collect_data.py
  run_inference.py
  record.sh
  inference.sh
  can_activate.sh
  find_all_can_port.sh

scripts/data_tools/
  standardize_dataset_actions.py
  validate_pose_dataset.py
  compare_pose_ik_qpos.py

scripts/diagnostics/
  check_obs.py
  check_camera_config.py
  test_obs_structure.py
  test_pose_tracking.py
  test_visualization.py
  verify_arm_vision.py
  verify_system.py
  viewenv.py
  view_frames.py
  analyze_frames.py

scripts/model_tools/
  check_model.py

scripts/archive/
  run_inference.py_FIXED_PART
```

The old script paths still work. For example, `python scripts/validate_pose_dataset.py ...` forwards to `scripts/data_tools/validate_pose_dataset.py`.

## Generated Outputs

```text
outputs/
  checkpoints/vision/
    final_vision_policy.pth
    loss_curve.png

  debug/
    debug_vision.jpg
    sapien_screenshot_0.png

  frames/
    viewenv/
    test_visualization/
```

New debug frames, screenshots, and model artifacts should be written to `outputs/` instead of the repository root.

## Tests

```text
test/
  check_obs.py
  test_can.py
  test_com_pose.py
  test_master.py
  test_pose.py
```

Keep quick checks that are part of test workflows in `test/`. One-off environment and camera diagnostics belong in `scripts/diagnostics/`.
