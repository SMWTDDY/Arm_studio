import numpy as np

class KeyboardTeleop:
    def __init__(self, env, step_size=0.005):
        self.env = env
        self.step_size = step_size
        self.gripper_state = -1.0 # 初始闭合
        self.action = np.zeros(7, dtype=np.float32)

    def get_action(self):
        self.action[:6] = 0.0 # 重置增量
        viewer = self.env.unwrapped.viewer
        
        if viewer is not None:
            window = viewer.window
            if window.key_down('i'): self.action[0] = self.step_size
            if window.key_down('k'): self.action[0] = -self.step_size
            if window.key_down('j'): self.action[1] = self.step_size
            if window.key_down('l'): self.action[1] = -self.step_size
            if window.key_down('u'): self.action[2] = self.step_size
            if window.key_down('o'): self.action[2] = -self.step_size
            
            if window.key_press('f'):
                self.gripper_state = -self.gripper_state
                
        self.action[6] = self.gripper_state
        return self.action.copy()