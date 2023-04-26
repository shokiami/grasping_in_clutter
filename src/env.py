import pybullet as p
import numpy as np
import time
import pybullet_data
import gym

NUM_JOINTS = 25

class FrankaPsyonic(gym.Env):
  def __init__(self):
    self.p = p
    self.p.connect(self.p.GUI)
    self.p.setGravity(0,0,-10)
    self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
    self.p.loadURDF("plane.urdf")
    startPos = [0,0,1]
    startOrientation = self.p.getQuaternionFromEuler([0,0,0])
    self.franka_psyonic_id = self.p.loadURDF("../urdfs/franka/panda.urdf", startPos, startOrientation, useFixedBase=True)

  def reset(self):
    default_joints = np.zeros(NUM_JOINTS)
    for i in range(NUM_JOINTS):
      self.p.resetJointState(self.franka_psyonic_id, i, default_joints[i])

  def step(self, action):
    for i in range(NUM_JOINTS):
      self.p.setJointMotorControl2(self.franka_psyonic_id, i, self.p.POSITION_CONTROL, targetPosition=action[i], force=10)
    self.p.stepSimulation()

if __name__ == '__main__':
  env = FrankaPsyonic()
  env.reset()
  while True:
    action = np.zeros(NUM_JOINTS)
    env.step(action)
    time.sleep(1./24.)
