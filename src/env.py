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
    self.p.loadURDF('plane.urdf')
    initPosition = [0, 0, 0]
    initOrientation = [0, 0, 0]
    self.franka_psyonic_id = self.p.loadURDF('../urdfs/franka/panda.urdf', initPosition, self.p.getQuaternionFromEuler(initOrientation), useFixedBase=True)
    self.createObject('duck', [0.5, 0, 0], [90, 0, 90], 1.0)

  def createObject(self, name, initPosition, initOrientation, mass):
    meshScale = [0.1, 0.1, 0.1]
    visualShapeId = self.p.createVisualShape(shapeType=self.p.GEOM_MESH,
                                        fileName=f'{name}.obj',
                                        rgbaColor=[1, 1, 1, 1],
                                        specularColor=[0.4, .4, 0],
                                        meshScale=meshScale)
    collisionShapeId = self.p.createCollisionShape(shapeType=self.p.GEOM_MESH,
                                              fileName=f'{name}_vhacd.obj',
                                              meshScale=meshScale)
    self.p.createMultiBody(baseMass=mass,  # kg
                          baseInertialFramePosition=[0, 0, 0],
                          baseCollisionShapeIndex=collisionShapeId,
                          baseVisualShapeIndex=visualShapeId,
                          basePosition=initPosition,
                          baseOrientation=self.p.getQuaternionFromEuler(initOrientation),
                          useMaximalCoordinates=True)


  def reset(self):
    default_joints = np.zeros(NUM_JOINTS)
    for i in range(NUM_JOINTS):
      self.p.resetJointState(self.franka_psyonic_id, i, default_joints[i])

  def step(self, action):
    target_position = action
    target_orientation = self.p.getQuaternionFromEuler([0, 0, 0])
    end_effector = 6
    joint_angles = self.p.calculateInverseKinematics(self.franka_psyonic_id, end_effector, target_position, target_orientation)
    for i in range(NUM_JOINTS):
      target_joint_angle = joint_angles[i] if i <= end_effector else 0
      self.p.setJointMotorControl2(self.franka_psyonic_id, i, self.p.POSITION_CONTROL, targetPosition=target_joint_angle, force=1000)
    self.p.stepSimulation()

if __name__ == '__main__':
  env = FrankaPsyonic()
  env.reset()
  while True:
    t = time.time()
    action = [0.4 * np.cos(t), 0.4 * np.sin(t), 0.8]
    env.step(action)
    time.sleep(1./120.)
