import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands = 2,
    min_detection_confidence=0.5,
) as hands:

    image = cv2.imread('hands.png')
    height, width, channels = image.shape
    image = cv2.flip(image, 1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    # HANDEDNESS
    # print("Handedness: ", results.multi_handedness)

    # HAND LANDMARKS
    # print("Hand landmarks: ", results.multi_hand_landmarks)

    if results.multi_hand_landmarks is not None:

        inde = [4 , 8, 12, 16, 20]
        for hand_landmarks in results.multi_hand_landmarks:
            for (i, points) in enumerate(hand_landmarks.landmark):
                if i in inde:
                    x = int(points.x * width)
                    y = int(points.y * height)
                    cv2.circle(image, (x, y), 1, (0, 0, 255), 4)

    image = cv2.flip(image, 1)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()