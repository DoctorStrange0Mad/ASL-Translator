import csv
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

file = open("asl_data.csv", "a", newline="")
writer = csv.writer(file)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb_frame)

        landmarks = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                    landmarks.append(lm.z)

        cv2.imshow("ASL Translator - Hand Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        # SAVING BLOCK MUST BE INSIDE LOOP
        if landmarks is not None:
            if key == ord('h'):
                writer.writerow(landmarks + ['HI'])
                file.flush()
                print("Saved HI")

            elif key == ord('b'):
                writer.writerow(landmarks + ['BYE'])
                file.flush()
                print("Saved BYE")

            elif key == ord('t'):
                writer.writerow(landmarks + ['THANKYOU'])
                file.flush()
                print("Saved THANKYOU")

            elif key == ord('o'):
                writer.writerow(landmarks + ['OK'])
                file.flush()
                print("Saved OK")

        if key == 27:
            break

file.close()
cap.release()
cv2.destroyAllWindows()
