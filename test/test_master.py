from pyAgxArm import create_agx_arm_config, AgxArmFactory

cfg = create_agx_arm_config(robot="piper", comm="can", channel="can_master")
robot = AgxArmFactory.create_arm(cfg)
robot.connect()

robot.set_follower_mode()