import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection  # 얼굴 검출을 위한 face_detection 모듈을 사용
mp_drawing = mp.solutions.drawing_utils  # 얼굴의 특징을 그리기 위한 drawing_utils 모듈을 사용
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)


def overlay(image, x, y, w, h, overlay_image):  # 대상 이미지 (3채널), x, y 좌표, width, height, 덮어씌울 이미지 (4채널)
    alpha = overlay_image[:, :, 3]  # BGRA
    mask_image = alpha / 255  # 0 ~ 255 -> 255 로 나누면 0 ~ 1 사이의 값 (1: 불투명, 0: 완전)


    for c in range(0, 3):  # channel BGR
        image[y - h:y + h, x - w:x + w, c] = (overlay_image[:, :, c] * mask_image) + (
                    image[y - h:y + h, x - w:x + w, c] * (1 - mask_image))

image_right_eye = cv2.imread('left-removebg-preview.png', cv2.IMREAD_UNCHANGED) # 100 x 100
image_left_eye = cv2.imread('right-removebg-preview.png', cv2.IMREAD_UNCHANGED) # 100 x 100
image_nose = cv2.imread('nose-removebg-preview.png', cv2.IMREAD_UNCHANGED) # 300 x 100 (가로, 세로)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)



        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:

            # 6개 특징 : 오른쪽 눈, 왼쪽 눈, 코 끝부분, 입 중심, 오른쪽 귀, 왼쪽 귀 (귀구슬점, 이주)
            for detection in results.detections:
                # mp_drawing.draw_detection(image, detection)
                # print(detection)

                # 특정 위치 가져오기
                keypoints = detection.location_data.relative_keypoints
                right_eye = keypoints[0]  # 오른쪽 눈
                left_eye = keypoints[1]  # 왼쪽 눈
                nose_tip = keypoints[2]  # 코 끝부분
                if right_eye.y < 0.2:
                    continue
                if left_eye.y < 0.2:
                    continue
                if right_eye.x > 0.8: # 화면 모서리에 걸치면
                     continue
                if left_eye.x < 0.2: # 화면 모서리에 걸치면
                     continue
                if nose_tip.y > 0.8: # 화면 모서리에 걸치면
                     continue

                h, w, _ = image.shape  # height, width, channel : 이미지로부터 세로, 가로 크기 가져옴
                right_eye = (int(right_eye.x * w) - 20, int(right_eye.y * h) - 100)  # 이미지 내에서 실제 좌표 (x, y)
                left_eye = (int(left_eye.x * w) + 20, int(left_eye.y * h) - 100)
                nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h))


                # image, x, y, w, h, overlay_image
                overlay(image, *right_eye, 50, 50, image_right_eye)
                overlay(image, *left_eye, 50, 50, image_left_eye)
                overlay(image, *nose_tip, 150, 50, image_nose)

            # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection', cv2.resize(image, None, fx=0.5, fy=0.5))
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()