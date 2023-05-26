import pybullet as p
import pybullet_data
import gym
import numpy as np

# env constants
SCALAR = 5.0
TABLE_HEIGHT = 0.58

# joint ids
DOF = 7
JOINTS = [0, 1, 2, 3, 4, 5, 6, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23]

# control gains
P_GAINS = [0.2] * len(JOINTS)
V_GAINS = [1.0] * len(JOINTS)

# obs/action space dims
OBS_SPACE = 22 + 3
ACT_SPACE = 17

class PsyonicPanda(gym.Env):
  def __init__(self, gui=True):
    # load env
    self.p = p
    if gui:
      self.p.connect(self.p.GUI)
    else:
      self.p.connect(self.p.DIRECT)
    self.p.setGravity(0, 0, -100)
    self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
    self.p.loadURDF('plane.urdf')

    # load robot
    self.robot = self.p.loadURDF('../urdfs/psyonic_panda.urdf', [0.0, 0.0, TABLE_HEIGHT * SCALAR], [0.0, 0.0, 0.0, 1.0], useFixedBase=True, globalScaling=SCALAR)

    # load amazon bin
    #self.pod = self.p.loadURDF('../urdfs/amazon_pod.urdf', [3.0, 3.0, 3.0], useFixedBase=True)

    self.table = self.p.loadURDF('../urdfs/table/table.urdf', [0.0, 0.5 * SCALAR, 0.0], self.p.getQuaternionFromEuler([0.0, 0.0, np.pi / 2]), globalScaling=SCALAR)
    self.cube = self.p.loadURDF('../urdfs/cube/cube.urdf', [0.0, 0.5 * SCALAR, (TABLE_HEIGHT + 0.1) * SCALAR], [0.0, 0.0, 0.0, 1.0], globalScaling=0.06*SCALAR)
    self.objs = [self.table, self.cube]

    # reset env
    self.reset()

  def reset(self):
    for i in range(DOF):
      joint_info = self.p.getJointInfo(self.robot, JOINTS[i])
      min_pos = joint_info[8]
      max_pos = joint_info[9]
      self.p.resetJointState(self.robot, JOINTS[i], (min_pos + max_pos) / 2.0)
    x = (0.8 * np.random.random() - 0.4) * SCALAR
    y = (0.7 * np.random.random() + 0.1) * SCALAR
    z = (TABLE_HEIGHT + 0.1) * SCALAR
    np.random.random() * SCALAR
    self.p.resetBasePositionAndOrientation(self.cube, [x, y, z], [0.0, 0.0, 0.0, 1.0])
    return self.getObservation()

  def state(self):
    return np.array(self.p.getLinkState(self.robot, DOF, computeForwardKinematics=True)[0])

  def goalState(self):
    return np.array(self.p.getBasePositionAndOrientation(self.cube)[0]) + np.array([0.0, 0.0, 0.5])

  def step(self, action):
    self.p.setJointMotorControlArray(self.robot, JOINTS, self.p.POSITION_CONTROL, targetPositions=action, positionGains=P_GAINS, velocityGains=V_GAINS)
    self.p.stepSimulation()

    s_prime = self.getObservation()
    dist = np.linalg.norm(self.state() - self.goalState())
    done = dist < 0.1
    r = -dist
    return s_prime, r, done

  def getObservation(self):
    return np.concatenate((self.getJointAngles(), self.getTouchData(), self.goalState()))

  def getJointAngles(self):
    joint_angles = np.array([joint_state[0] for joint_state in self.p.getJointStates(self.robot, JOINTS)])
    return joint_angles

  def getTouchData(self):
    touch_data = np.zeros(5)
    for i in range(5):
      for obj in self.objs:
        contact_points = self.p.getContactPoints(bodyA=self.robot, bodyB=obj, linkIndexA=11 + 3 * i)
        for contact_point in contact_points:
          normal_force = contact_point[9]
          touch_data[i] += normal_force
    return touch_data
  
  def inverseKinematics(self, target_pos, target_ori):
    joint_angles = list(self.p.calculateInverseKinematics(self.robot, DOF, target_pos, target_ori))[:DOF]
    return joint_angles + [0.0] * (len(JOINTS) - DOF)
