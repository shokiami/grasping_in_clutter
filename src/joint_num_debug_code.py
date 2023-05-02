import pybullet as p
import numpy as np
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")

startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
# handid = p.loadURDF("ability-hand-api/URDF/ability_hand_right.urdf", [0,0,0.5], [0,0,0,1], useFixedBase=True, globalScaling=1)
#arm_hand = p.loadURDF("xarm/xarm5_robot_psyonic.urdf", [0,0,0.3], [0,0,0,1], useFixedBase=True, globalScaling=1)
arm_hand = p.loadURDF("../urdfs/franka/panda.urdf", [0,0,0.3], [0,0,0,1], useFixedBase=True, globalScaling=1)
# p.changeDynamics(handid, -1, mass=10)

# control the hand
print(p.getNumJoints(arm_hand))

for i in range (10000):
    # p.setJointMotorControl2(handid, )
    random_rot = np.random.uniform(-10.57, 10.57)
    p.setJointMotorControl2(arm_hand, 1, p.POSITION_CONTROL, targetPosition=random_rot, force =100)
    p.stepSimulation()
    time.sleep(1./24.)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
p.disconnect()
