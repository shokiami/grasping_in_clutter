import pybullet as p
import time
import pybullet_data
import gym

class PsyonicPanda(gym.Env):
  def __init__(self):
    # load env
    self.p = p
    self.p.connect(self.p.GUI)
    self.p.setGravity(0, 0, -9.81)
    self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
    self.p.loadURDF('plane.urdf')

    # load robot
    self.dof = 7
    self.kp = 0.1
    self.kd = 1.0
    self.max_torque = 100
    self.robot = self.p.loadURDF('../urdfs/psyonic_panda.urdf', [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], useFixedBase=True)
    self.num_joints = self.p.getNumJoints(self.robot)
    print(f'num_joints: {self.num_joints}')

    # load duck
    self.duck = self.createObject('duck', [0.5, 0, 0], [90, 0, 90], 1.0)

    # reset robot
    self.reset()

  def createObject(self, name, initPosition, init_ori, mass):
    meshScale = [0.1, 0.1, 0.1]
    visualShapeId = self.p.createVisualShape(shapeType=self.p.GEOM_MESH,
                                            fileName=f'{name}.obj',
                                            rgbaColor=[1, 1, 1, 1],
                                            specularColor=[0.4, .4, 0],
                                            meshScale=meshScale)
    collisionShapeId = self.p.createCollisionShape(shapeType=self.p.GEOM_MESH,
                                                  fileName=f'{name}_vhacd.obj',
                                                  meshScale=meshScale)
    return self.p.createMultiBody(baseMass=mass, # kg
                                  baseInertialFramePosition=[0, 0, 0],
                                  baseCollisionShapeIndex=collisionShapeId,
                                  baseVisualShapeIndex=visualShapeId,
                                  basePosition=initPosition,
                                  baseOrientation=init_ori,
                                  useMaximalCoordinates=True)

  def reset(self):
    for i in range(self.dof):
      joint_info = self.p.getJointInfo(self.robot, i)
      min_pos = joint_info[8]
      max_pos = joint_info[9]
      self.p.resetJointState(self.robot, i, (min_pos + max_pos) / 2.0)

  def step(self, action):
    target_pos, target_ori = action
    joint_angles = list(self.p.calculateInverseKinematics(self.robot, self.dof, target_pos, target_ori))
    for i in range(self.dof):
      self.p.setJointMotorControl2(bodyUniqueId=self.robot,
                                  jointIndex=i,
                                  controlMode=self.p.POSITION_CONTROL,
                                  targetPosition=joint_angles[i],
                                  force=self.max_torque,
                                  positionGain=self.kp,
                                  velocityGain=self.kd)
    self.p.stepSimulation()

if __name__ == '__main__':
  env = PsyonicPanda()
  env.reset()
  start_t = time.time()
  while True:
    target_pos = list(env.p.getBasePositionAndOrientation(env.duck)[0])
    target_pos[2] += 0.5
    target_ori = [1.0, 0.0, 0.0, 0.0] # downward
    action = (target_pos, target_ori)
    env.step(action)
    time.sleep(1./120.)
