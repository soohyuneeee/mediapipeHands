import math

import cv2
#미디어파이프 임포트
import mediapipe as mp
#딜레이 시간
FRAME_DELAY = 100

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_hands = mp.solutions.hands
mp_fingers = mp_hands.HandLandmark

num = 0
def get_dist(p1,p2,p3):
    aplustb = math.sqrt(p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2
    print(math.sqrt(p3.x-p2.x)**2, aplustb)

def get_angle(ps_x,ps_y, p1_x, p1_y, p2_x,p2_y):
    # print(ps,p1,p2)
    angle1 = abs(math.atan((p1_y-ps_y)/(p1_x-ps_x)))
    angle2 = abs(math.atan((p2_y-ps_y)/(p2_x-ps_x)))
    angle = abs(angle1-angle2) * 180 / math.pi
    # print(f'angle: {angle}')
    if angle > 20:
        return True

# def isfold_y(nowtip,nowdip,nowmcp,wrist):
#     if wrist < nowtip and wrist < nowdip and wrist < nowmcp:
#         if nowdip > nowtip:
#             return True
#         else:
#             return False
#     else:
#         if nowtip > nowmcp and nowdip > nowmcp:
#             return True
#         else:
#             return False

def run():
    cap = cv2.VideoCapture(0)

    hands = mp_hands.Hands(
        max_num_hands=5,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print('Ignoring empty camera frame.')
            continue
        image = cv2.flip(image, 1)

        image.flags.writeable=False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #관절에 대한 정보
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        width, height, _ = image.shape

        #result 후처리 과정
        if results.multi_hand_landmarks:
            #손의 개수 출력
            #print(len(results.multi_hand_landmarks))
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[mp_fingers.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_fingers.INDEX_FINGER_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_fingers.MIDDLE_FINGER_TIP]
                ring_finger_tip = hand_landmarks.landmark[mp_fingers.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_fingers.PINKY_TIP]
                # 손 끝에서 두번째 = 7,11,15,19
                thumb_ip = hand_landmarks.landmark[mp_fingers.THUMB_IP]
                index_finger_dip = hand_landmarks.landmark[mp_fingers.INDEX_FINGER_DIP]
                middle_finger_dip = hand_landmarks.landmark[mp_fingers.MIDDLE_FINGER_DIP]
                ring_finger_dip = hand_landmarks.landmark[mp_fingers.RING_FINGER_DIP]
                pinky_dip = hand_landmarks.landmark[mp_fingers.PINKY_DIP]
                # 손 끝에서 세번째 = 6,10,14,18
                thumb_mcp = hand_landmarks.landmark[mp_fingers.THUMB_MCP]
                index_finger_pip = hand_landmarks.landmark[mp_fingers.INDEX_FINGER_PIP]
                middle_finger_pip = hand_landmarks.landmark[mp_fingers.MIDDLE_FINGER_PIP]
                ring_finger_pip = hand_landmarks.landmark[mp_fingers.RING_FINGER_PIP]
                pinky_pip = hand_landmarks.landmark[mp_fingers.PINKY_PIP]
                # 마디 5,9,13,17
                thumb_cmc = hand_landmarks.landmark[mp_fingers.THUMB_CMC]
                index_finger_mcp = hand_landmarks.landmark[mp_fingers.INDEX_FINGER_MCP]
                middle_finger_mcp = hand_landmarks.landmark[mp_fingers.MIDDLE_FINGER_MCP]
                ring_finger_mcp = hand_landmarks.landmark[mp_fingers.RING_FINGER_MCP]
                pinky_mcp = hand_landmarks.landmark[mp_fingers.PINKY_MCP]
                # 손목(중점)
                wrist = hand_landmarks.landmark[mp_fingers.WRIST].y
                tips = []
                tips.extend([thumb_tip.y, index_finger_tip.y, middle_finger_tip.y, ring_finger_tip.y, pinky_tip.y])
                dips_y = []
                dips_y.extend([thumb_ip.y, index_finger_dip.y, middle_finger_dip.y, ring_finger_dip.y, pinky_dip.y])
                dips_x = []
                dips_x.extend([thumb_ip.x, index_finger_dip.x, middle_finger_dip.x, ring_finger_dip.x, pinky_dip.x])
                pips_y = []
                pips_y.extend([thumb_mcp.y, index_finger_pip.y, middle_finger_pip.y, ring_finger_pip.y, pinky_pip.y])
                pips_x = []
                pips_x.extend([thumb_mcp.x, index_finger_pip.x, middle_finger_pip.x, ring_finger_pip.x, pinky_pip.x])
                mcp_y = []
                mcp_y.extend([thumb_cmc.y, index_finger_mcp.y, middle_finger_mcp.y, ring_finger_mcp.y, pinky_mcp.y])
                mcp_x = []
                mcp_x.extend([thumb_cmc.x, index_finger_mcp.x, middle_finger_mcp.x, ring_finger_mcp.x, pinky_mcp.x])
                # get_dist()
                num=0
                for i in range(0,5):
                    if get_angle(pips_x[i],pips_y[i], dips_x[i],dips_y[i], mcp_x[i],mcp_y[i])==True:
                        num += 1
                print(num)

                # get_angle(hand_landmarks.landmark[mp_fingers.INDEX_FINGER_PIP],
                #           hand_landmarks.landmark[mp_fingers.INDEX_FINGER_DIP],
                #           hand_landmarks.landmark[mp_fingers.INDEX_FINGER_MCP])
                cv2.putText(
                    image,
                    text=f'{str(int(index_finger_tip.x * width))},{str(int(index_finger_tip.y * height))}',
                    org=(100,100),
                    fontFace =cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255,255,255),
                    thickness=2
                )
                mp_drawing.draw_landmarks(
                    image,
                    #각 손의 정보
                    hand_landmarks,
                    #손가락 연결
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )



                # num=0
                # for i in range(0,5):
                #     if isfold_y(tips[i], dips[i], mcp[i],wrist)==False:
                #         num += 1
                # print(num)




                # 손 끝 = 4, 8, 12, 16, 20


                #index_finger_tip = hand_landmarks.landmark[mp_fingers.INDEX_FINGER_TIP]
                #print(f'x: {index_finger_tip.x},\ny: {index_finger_tip.y}')
                #print("-------")
        cv2.imshow('MediaPipe Hands', image)
        cv2.waitKey(FRAME_DELAY)
    cap.release()




run()


# class HandLandmark(enum.IntEnum):
#   """The 21 hand landmarks."""
#   WRIST = 0
#   THUMB_CMC = 1
#   THUMB_MCP = 2
#   THUMB_IP = 3
#   THUMB_TIP = 4
#   INDEX_FINGER_MCP = 5
#   INDEX_FINGER_PIP = 6
#   INDEX_FINGER_DIP = 7
#   INDEX_FINGER_TIP = 8
#   MIDDLE_FINGER_MCP = 9
#   MIDDLE_FINGER_PIP = 10
#   MIDDLE_FINGER_DIP = 11
#   MIDDLE_FINGER_TIP = 12
#   RING_FINGER_MCP = 13
#   RING_FINGER_PIP = 14
#   RING_FINGER_DIP = 15
#   RING_FINGER_TIP = 16
#   PINKY_MCP = 17
#   PINKY_PIP = 18
#   PINKY_DIP = 19
#   PINKY_TIP = 20
