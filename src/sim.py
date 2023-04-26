import pybullet as p
import time
import pybullet_data

physicsClient = p.connect(p.GUI)
p.setGravity(0,0,-10)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
arm_hand = p.loadURDF("../urdfs/franka/panda.urdf", startPos, startOrientation, useFixedBase=True)

for i in range (10000):
    random_rot = 5
    p.setJointMotorControl2(arm_hand, 2, p.POSITION_CONTROL, targetPosition=random_rot, force=100)
    p.stepSimulation()
    time.sleep(1./24.)

p.disconnect()
