import pybullet as p
import time
import pybullet_data
import gym
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

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
    self.kp = 0.3
    self.kd = 1.0
    self.robot = self.p.loadURDF('../urdfs/psyonic_panda.urdf', [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], useFixedBase=True)
    self.num_joints = self.p.getNumJoints(self.robot)
    print(f'num_joints: {self.num_joints}')

    # load duck
    self.duck = self.createObject('duck', [0.0, 0.5, 0.0], [1.0, 1.0, 1.0, 1.0], 1.0)

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
    target_pos, hand_msg = action
    target_ori = [1.0, 0.0, 0.0, 0.0]
    self.control_arm(target_pos, target_ori)
    self.control_hand(hand_msg)
    self.p.stepSimulation()

  def control_arm(self, target_pos, target_ori):
    joint_angles = self.p.calculateInverseKinematics(self.robot, self.dof, target_pos, target_ori)
    for i in range(self.dof):
      self.p.setJointMotorControl2(bodyUniqueId=self.robot,
                                  jointIndex=i,
                                  controlMode=self.p.POSITION_CONTROL,
                                  targetPosition=joint_angles[i],
                                  positionGain=self.kp,
                                  velocityGain=self.kd)

  def control_hand(self, hand_msg):
    # control fingers
    for i in range(10, 21):
      if i % 3 == 0:
        continue
      joint_angle = hand_msg[(i - 10) // 3]
      self.p.setJointMotorControl2(bodyUniqueId=self.robot,
                                  jointIndex=i,
                                  controlMode=self.p.POSITION_CONTROL,
                                  targetPosition=joint_angle,
                                  positionGain=self.kp,
                                  velocityGain=self.kd)
    # control thumb
    for i in range(22, 24):
      joint_angle = hand_msg[27 - i]
      self.p.setJointMotorControl2(bodyUniqueId=self.robot,
                                  jointIndex=i,
                                  controlMode=self.p.POSITION_CONTROL,
                                  targetPosition=joint_angle,
                                  positionGain=self.kp,
                                  velocityGain=self.kd)

def getPose(frame):
  # run hand tracker
  frame.flags.writeable = False
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = hands.process(frame)
  frame.flags.writeable = True
  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

  # extract hand pose
  pose = None
  if results.multi_hand_landmarks:
    for hand, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
      if hand.classification[0].label == 'Left':
        pose = hand_landmarks.landmark
        mp_drawing.draw_landmarks(
          frame,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

  # flip frame for selfie-view
  return pose, cv2.flip(frame, 1)

def getHandMsg(pose):
  hand_msg = [0.0] * 6

  # scalar for standardizing hand size
  z = 4.0 / (dist(pose[0], pose[5]) + dist(pose[0], pose[9]) + dist(pose[0], pose[13]) + dist(pose[0], pose[17]))

  JOINT_MIN = 0.0
  JOINT_MAX = 2.0
  hand_msg[0] = interpolate(z * dist(pose[0], pose[8]), 1.9, 0.9, JOINT_MIN, JOINT_MAX) # index finger
  hand_msg[1] = interpolate(z * dist(pose[0], pose[12]), 2.0, 0.8, JOINT_MIN, JOINT_MAX) # middle finger
  hand_msg[2] = interpolate(z * dist(pose[0], pose[16]), 1.9, 0.7, JOINT_MIN, JOINT_MAX) # ring finger
  hand_msg[3] = interpolate(z * dist(pose[0], pose[20]), 1.8, 0.7, JOINT_MIN, JOINT_MAX) # pinky finger
  hand_msg[4] = interpolate(z * dist(pose[0], pose[4]), 1.2, 0.9, JOINT_MIN, JOINT_MAX) # thumb
  hand_msg[5] = -interpolate(z * dist(pose[2], pose[17]), 0.9, 0.6, JOINT_MIN, JOINT_MAX) # thumb rotator

  return hand_msg

def getHandXYZ(pose):
  MIN_X = -1.0
  MAX_X = 1.0
  MIN_Y = 0.0
  MAX_Y = 1.0
  MIN_Z = 0.0
  MAX_Z = 1.0

  # get hand x and y
  x = interpolate(pose[0].x, 1.0, 0.0, MIN_X, MAX_X)
  y = interpolate(pose[0].y, 1.0, 0.0, MIN_Y, MAX_Y)

  # get hand z
  z = 4.0 / (dist(pose[0], pose[5]) + dist(pose[0], pose[9]) + dist(pose[0], pose[13]) + dist(pose[0], pose[17]))
  z = interpolate(z, 3.0, 10.0, MIN_Z, MAX_Z)

  return x, y, z

def interpolate(val, real_min, real_max, target_min, target_max):
  p = (val - real_min) / (real_max - real_min)
  new_val = (1 - p) * target_min + p * target_max
  return min(max(new_val, target_min), target_max)

def dist(landmark_1, landmark_2):
  dx = landmark_2.x - landmark_1.x
  dy = landmark_2.y - landmark_1.y
  return (dx * dx + dy * dy)**0.5

if __name__ == '__main__':
  env = PsyonicPanda()
  env.reset()
  start_t = time.time()
  cap = cv2.VideoCapture(0)
  target_pos = env.p.getBasePositionAndOrientation(env.robot)[0]
  hand_msg = [0.0] * 6

  with mp_hands.Hands(
      model_complexity=1,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
      success, frame = cap.read()
      if not success:
        print('empty camera frame')
        continue
    
      # get right-hand pose from mediapipe
      pose, frame = getPose(frame)

      # get message from pose
      if pose:
        target_pos = getHandXYZ(pose)
        hand_msg = getHandMsg(pose)

      action = (target_pos, hand_msg)
      env.step(action)
      time.sleep(1./120.)

      cv2.imshow('MediaPipe', frame)
      key = cv2.waitKey(1)
      if key & 0xFF == ord('q'):
        break

  cap.release()
  env.p.disconnect()
