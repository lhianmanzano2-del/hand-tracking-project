import cv2
import mediapipe as mp
import time


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()


# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# FPS
# ============================================================

previous_time = 0


# ============================================================
# FINGER COUNT FUNCTION
# ============================================================

def count_fingers(hand_landmarks, hand_label):

    fingers = 0

    # --------------------------------------------------------
    # THUMB
    # --------------------------------------------------------

    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]

    if hand_label == "Right":
        if thumb_tip.x < thumb_ip.x:
            fingers += 1
    else:
        if thumb_tip.x > thumb_ip.x:
            fingers += 1


    # --------------------------------------------------------
    # INDEX FINGER
    # --------------------------------------------------------

    if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
        fingers += 1


    # --------------------------------------------------------
    # MIDDLE FINGER
    # --------------------------------------------------------

    if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
        fingers += 1


    # --------------------------------------------------------
    # RING FINGER
    # --------------------------------------------------------

    if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:
        fingers += 1


    # --------------------------------------------------------
    # PINKY
    # --------------------------------------------------------

    if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:
        fingers += 1


    return fingers


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read camera.")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Process frame
    result = hands.process(rgb)


    # ========================================================
    # HAND DETECTED
    # ========================================================

    if result.multi_hand_landmarks:

        for hand_landmarks, handedness in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):

            hand_label = handedness.classification[0].label


            # =================================================
            # GET HAND LANDMARK POSITIONS
            # =================================================

            h, w, _ = frame.shape

            points = []

            for landmark in hand_landmarks.landmark:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                points.append((x, y))


            # =================================================
            # BOUNDING BOX
            # =================================================

            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]

            x_min = min(x_values) - 20
            x_max = max(x_values) + 20

            y_min = min(y_values) - 20
            y_max = max(y_values) + 20


            # Keep box inside screen
            x_min = max(0, x_min)
            y_min = max(0, y_min)

            x_max = min(w, x_max)
            y_max = min(h, y_max)


            # =================================================
            # COUNT FINGERS
            # =================================================

            finger_count = count_fingers(
                hand_landmarks,
                hand_label
            )


            # =================================================
            # DRAW CONNECTIONS
            # =================================================

            connections = [
                # Thumb
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),

                # Index
                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),

                # Middle
                (0, 9),
                (9, 10),
                (10, 11),
                (11, 12),

                # Ring
                (0, 13),
                (13, 14),
                (14, 15),
                (15, 16),

                # Pinky
                (0, 17),
                (17, 18),
                (18, 19),
                (19, 20),

                # Palm
                (5, 9),
                (9, 13),
                (13, 17)
            ]


            # Draw cyan lines
            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 255, 0),
                    3
                )


            # =================================================
            # DRAW LANDMARK POINTS
            # =================================================

            for point in points:

                cv2.circle(
                    frame,
                    point,
                    6,
                    (255, 255, 0),
                    -1
                )


            # =================================================
            # DRAW BOUNDING BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 255),
                2
            )


            # =================================================
            # HAND LABEL
            # =================================================

            cv2.putText(
                frame,
                hand_label,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )


            # =================================================
            # LARGE FINGER NUMBER
            # =================================================

            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2


            cv2.putText(
                frame,
                str(finger_count),
                (center_x - 25, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 255, 255),
                7
            )


            # =================================================
            # FINGER TEXT
            # =================================================

            cv2.putText(
                frame,
                f"Fingers: {finger_count}",
                (x_min, y_max + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


    # ========================================================
    # FPS CALCULATION
    # ========================================================

    current_time = time.time()

    fps = 1 / (current_time - previous_time) \
        if previous_time != 0 else 0

    previous_time = current_time


    # ========================================================
    # DISPLAY FPS
    # ========================================================

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "MEDIAPIPE HAND TRACKER",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "MediaPipe Hand Tracker",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEAN UP
# ============================================================

cap.release()
cv2.destroyAllWindows()
hands.close()

print("Hand tracker closed.")
