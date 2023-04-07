import serial
from serial.tools import list_ports
import struct
import platform
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Search for Serial Port to use
def initSerial():
  print('Searching for serial ports...')
  com_ports_list = list(list_ports.comports())
  port = ''

  for p in com_ports_list:
    if(p):
      if platform.system() == 'Linux' or platform.system() == 'Darwin':
        if 'USB' in p[0]:
          port = p
          print('Found:', port)
          break
      elif platform.system() == 'Windows':
        if 'COM' in p[0]:
          port = p
          print('Found:', port)
          break
  if not port:
    print('No port found')
    quit()
    
  try: 
    print('Connecting...')
    ser = serial.Serial(port[0], 460800, timeout = 0.02)
    print('Connected!')
  except: 
    print('Failed to Connect!')
    exit()

  #Clear Buffer to start
  ser.reset_input_buffer()
    
  return ser

# Generate Message to send to hand from array of positions (floating point)
def generateTX(positions):
  txBuf = []
  
  # Address in byte 0
  txBuf.append((struct.pack('<B', 0x50))[0])
  
  # Format Header in byte 1
  txBuf.append((struct.pack('<B', 0x10))[0])
  
  # Position data for all 6 fingers, scaled to fixed point representation
  for i in range(0,6):
    posFixed = int(positions[i] * 32767 / 150)
    txBuf.append((struct.pack('<B',(posFixed & 0xFF)))[0])
    txBuf.append((struct.pack('<B',(posFixed >> 8) & 0xFF))[0])
  
  # calculate checksum
  cksum = 0
  for b in txBuf:
    cksum = cksum + b
  cksum = (-cksum) & 0xFF
  txBuf.append((struct.pack('<B', cksum))[0])
  
  return txBuf

def getPose(image):
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  results = hands.process(image)

  # Draw the hand annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  pose = None
  if results.multi_hand_landmarks:
    for hand, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
      if hand.classification[0].label == 'Left':
        pose = hand_landmarks.landmark
        mp_drawing.draw_landmarks(
          image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

  # Flip the image horizontally for a selfie-view display.
  return pose, cv2.flip(image, 1)


def getMsg(pose):
  MIN = 5
  MAX = 90
  THUMB_MAX = 50

  msg = [15.0] * 6

  # standardize hand size
  scalar = 4.0 / (dist(pose[0], pose[5]) + dist(pose[0], pose[9]) + dist(pose[0], pose[13]) + dist(pose[0], pose[17]))

  # index finger
  msg[0] = scale(scalar * dist(pose[0], pose[8]), 0.9, 1.9, MIN, MAX)

  # middle finger
  msg[1] = scale(scalar * dist(pose[0], pose[12]), 0.8, 2.0, MIN, MAX)

  # ring finger
  msg[2] = scale(scalar * dist(pose[0], pose[16]), 0.7, 1.9, MIN, MAX)

  # pinky finger
  msg[3] = scale(scalar * dist(pose[0], pose[20]), 0.7, 1.8, MIN, MAX)

  # thumb
  msg[4] = scale(scalar * dist(pose[0], pose[4]), 0.9, 1.2, MIN, THUMB_MAX)

  # thumb rotator
  msg[5] = -scale(scalar * dist(pose[2], pose[17]), 0.6, 0.9, MIN, THUMB_MAX)

  return msg

def scale(dist, real_min, real_max, control_min, control_max):
  p = 1 - (dist - real_min) / (real_max - real_min)
  control = p * (control_max - control_min) + control_min
  return min(max(control, control_min), control_max)

def dist(landmark_1, landmark_2):
  dx = landmark_2.x - landmark_1.x
  dy = landmark_2.y - landmark_1.y
  # dz = landmark_2.z - landmark_1.z
  return (dx * dx + dy * dy)**0.5

if __name__ == '__main__':
  ser = initSerial()
  cap = cv2.VideoCapture(0)
  msg = [15.0] * 6
  with mp_hands.Hands(
      model_complexity=1,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print('empty camera frame')
        continue

      # Get right-hand pose from mediapipe
      pose, image = getPose(image)

      # Get message from pose
      if pose:
        msg = getMsg(pose)

      # Send message to psyonic hand
      ser.write(generateTX(msg))

      # Read first response byte
      data = ser.read(1)

      cv2.imshow('MediaPipe', image)
      if cv2.waitKey(1) & 0xFF == 27:
        break

  cap.release()
  ser.close()		
