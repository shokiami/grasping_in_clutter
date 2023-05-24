from env import PsyonicPanda, SCALAR, TABLE_HEIGHT
import time
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

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
  MIN_X = -SCALAR
  MAX_X = SCALAR
  MIN_Y = 0.0
  MAX_Y = SCALAR
  MIN_Z = 0.0
  MAX_Z = SCALAR

  # get hand x and y
  x = interpolate(pose[0].x, 1.0, 0.0, MIN_X, MAX_X)
  y = interpolate(pose[0].y, 1.0, 0.0, MIN_Y, MAX_Y)

  # get hand z
  z = 4.0 / (dist(pose[0], pose[5]) + dist(pose[0], pose[9]) + dist(pose[0], pose[13]) + dist(pose[0], pose[17]))
  z = interpolate(z, 3.0, 10.0, MIN_Z, MAX_Z)

  return x, y, z + TABLE_HEIGHT * SCALAR

def interpolate(val, real_min, real_max, target_min, target_max):
  p = (val - real_min) / (real_max - real_min)
  new_val = (1 - p) * target_min + p * target_max
  return min(max(new_val, target_min), target_max)

def dist(landmark_1, landmark_2):
  dx = landmark_2.x - landmark_1.x
  dy = landmark_2.y - landmark_1.y
  return (dx * dx + dy * dy)**0.5

def computeJointAngles(env, target_pos, target_ori, hand_msg):
  # add arm joint angles
  joint_angles = env.inverseKinematics(target_pos, target_ori)

  # add finger joint angles
  for i in range(4):
    joint_angles[2 * i + 7] = hand_msg[i]
    joint_angles[2 * i + 8] = hand_msg[i]
  
  # add thumb joint angles
  joint_angles[15] = hand_msg[5]
  joint_angles[16] = hand_msg[4]

  return joint_angles

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

      target_ori = env.p.getQuaternionFromEuler([-np.pi / 2, 0.0, 0.0])
      action = computeJointAngles(env, target_pos, target_ori, hand_msg)
      env.step(action)
      time.sleep(1./240.)

      cv2.imshow('MediaPipe', frame)
      key = cv2.waitKey(1)
      if key & 0xFF == ord('q'):
        break

  cap.release()
  env.p.disconnect()
