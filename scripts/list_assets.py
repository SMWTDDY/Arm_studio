import gymnasium as gym
import mani_skill.envs
import sys

def main():
    print("\n" + "="*50)
    print("      ManiSkill 已注册环境列表")
    print("="*50)
    
    # 获取所有注册的环境名
    all_envs = list(gym.registry.keys())
    # 过滤出包含 ManiSkill 的环境
    ms_envs = [env for env in all_envs if "mani_skill" in env.lower() or "Piper" in env]
    
    # 分门别类显示
    categories = {
        "基础任务": ["PickCube", "PushCube", "StackCube"],
        "交互任务": ["OpenCabinet", "CloseCabinet", "PushChair"],
        "视觉任务": ["PandaPointcloud", "Anymal"],
        "自定义任务": ["Piper"]
    }

    for cat, keywords in categories.items():
        print(f"\n[{cat}]")
        found = False
        for env_id in sorted(ms_envs):
            if any(kw in env_id for kw in keywords):
                print(f"  - {env_id}")
                found = True
        if not found:
            print("  (暂无该分类下的环境)")

    print("\n" + "="*50)
    print("使用提示:")
    print("1. 如果想使用 YCB 物品，请确保本地存在 ~/.mani_skill/data/assets 目录")
    print("2. 您可以直接在 viewenv.py 中尝试运行上述任意环境名")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
