import os
import sys
import ast
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_factory.config.manager import ConfigManager


def verify_config_migration():
    base_cfg = OmegaConf.create(
        {
            "agent_sp": {
                "iters": 1,
                "base_policy": {"type": "", "ckpt_path": "", "base_config_path": ""},
            },
            "actor": {"type": "dsrl_policy"},
        }
    )
    user_cfg = OmegaConf.create(
        {
            "actor": {
                "base_policy": {
                    "agent_type": "Diffusion_ITQC",
                    "checkpoint_path": "/tmp/base.pth",
                }
            }
        }
    )
    merged = ConfigManager.merge_configs(base_cfg, user_cfg)
    bp = merged.agent_sp.base_policy
    assert bp.type == "Diffusion_ITQC", "agent_type -> type migration failed"
    assert bp.ckpt_path == "/tmp/base.pth", "checkpoint_path -> ckpt_path migration failed"
    assert hasattr(bp, "base_config_path"), "base_config_path missing after migration"
    print("✅ config migration check passed")


def verify_sample_action_signatures():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    targets = [
        os.path.join(repo_root, "agent_factory/agents/mixins/actor/diffusion.py"),
        os.path.join(repo_root, "agent_factory/agents/mixins/actor/conditional_diffusion.py"),
    ]
    for path in targets:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "sample_action"]
        assert funcs, f"sample_action not found in {path}"
        # Use the first sample_action in file (the mixin method)
        arg_names = [a.arg for a in funcs[0].args.args]
        assert "initial_noise" in arg_names, f"{path} sample_action missing initial_noise"
    print("✅ sample_action(initial_noise=...) interface check passed")


def main():
    verify_config_migration()
    verify_sample_action_signatures()
    print("🎉 DSRL plugin contract verification passed")


if __name__ == "__main__":
    main()
