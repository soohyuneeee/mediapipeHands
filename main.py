import cv2

FRAME_DELAY = 200


def run():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print('Ignoring empty camera frame.')
            continue

        cv2.imshow('MediaPipe Hands', image)
        cv2.waitKey(FRAME_DELAY)
    cap.release()


run()
