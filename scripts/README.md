# Scripts

The root of `scripts/` contains the main user-facing entry points:

```text
collect_data.py
run_inference.py
record.sh
inference.sh
can_activate.sh
find_all_can_port.sh
```

Specialized tools are grouped by purpose:

```text
data_tools/     dataset conversion, validation, and IK comparison
diagnostics/    environment, camera, rendering, and pose tracking checks
model_tools/    checkpoint inspection
archive/        old fragments kept for reference
```

Compatibility wrappers remain at the old paths, so existing commands such as `python scripts/validate_pose_dataset.py ...` continue to work.
