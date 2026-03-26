import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory

cfg = create_agx_arm_config(robot="piper", comm="can", channel="can_master")
robot = AgxArmFactory.create_arm(cfg)
robot.connect()

time.sleep(0.5)
print("robotic arm is_ok =", robot.is_ok())