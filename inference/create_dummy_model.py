import torch
import os
model_path = "policy.pi0"
state_dict = {
        'obs_dim': 16,
        'action_dim': 7,
        'model': {
            '0.weight': torch.randn(256, 16),         
            '0.bias': torch.randn(256),
            '2.weight': torch.randn(256, 256),
            '2.bias': torch.randn(256),
            '4.weight': torch.randn(7, 256),
            '4.bias': torch.randn(7),
       }
    }
torch.save(state_dict, model_path)
print(f"模型已更新，确保输出维度为 7: {os.path.abspath(model_path)}")