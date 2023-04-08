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

def extractData(ser):
  pos_data = [0.0] * 6
  touch_data = [0.0] * 30

  data = ser.read(1)
  if len(data) != 1:
    return pos_data, touch_data

  replyFormat = data[0]
  if (replyFormat & 0xF) == 2:
    replyLen = 38
  else:
    replyLen = 71
  data = ser.read(replyLen)

  if len(data) == replyLen:

    # Extract position data
    for i in range(6):
      rawData = struct.unpack('<h', data[i*4:2+(i*4)])[0]
      pos_data[i] = rawData * 150 / 32767
    pos_data[5] = -pos_data[5]

    # Extract touch data
    if replyLen == 71:
      for i in range(15):
        dualData = data[(i*3)+24:((i+1)*3)+24]
        data1 = struct.unpack('<H', dualData[0:2])[0] & 0x0FFF
        data2 = (struct.unpack('<H', dualData[1:3])[0] & 0xFFF0) >> 4
        touch_data[i*2] = int(data1)
        touch_data[(i*2)+1] = int(data2)

  return pos_data, touch_data

def getPose(frame):
  # To improve performance, optionally mark the frame as not writeable to
  # pass by reference.
  frame.flags.writeable = False
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = hands.process(frame)

  # Draw the hand annotations on the frame.
  frame.flags.writeable = True
  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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

  # Flip the frame horizontally for a selfie-view display.
  return pose, cv2.flip(frame, 1)

def getMsg(pose):
  MIN = 0
  MAX = 100
  THUMB_MAX = 60

  msg = [15.0] * 6

  # standardize hand size
  scalar = 4.0 / (dist(pose[0], pose[5]) + dist(pose[0], pose[9])
                  + dist(pose[0], pose[13]) + dist(pose[0], pose[17]))

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
      success, frame = cap.read()
      if not success:
        print('empty camera frame')
        continue

      # Get right-hand pose from mediapipe
      pose, frame = getPose(frame)

      # Get message from pose
      if pose:
        msg = getMsg(pose)

      # Send message to psyonic hand
      ser.write(generateTX(msg))

      # Read first response byte
      pos_data, touch_data = extractData(ser)

      cv2.imshow('MediaPipe', frame)
      if cv2.waitKey(1) & 0xFF == 27:
        break

  cap.release()
  ser.close()		
