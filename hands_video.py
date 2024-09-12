import cv2
import mediapipe as mp

# Inicializamos MediaPipe para la segmentación y detección de manos
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inicializamos la segmentación de selfies
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Configuramos la detección de manos
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Procesamos el frame
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Aplicamos segmentación para detectar la persona
        segmentation_result = selfie_segmentation.process(frame_rgb)

        # Creamos una máscara binaria
        mask = segmentation_result.segmentation_mask
        condition = mask > 0.5

        # Aplicamos desenfoque al fondo
        blurred_frame = cv2.GaussianBlur(frame, (55, 55), 0)

        # Combinar el fondo desenfocado con la persona
        output_frame = frame.copy()
        output_frame[~condition] = blurred_frame[~condition]

        # Detección de manos y dibujo
        results = hands.process(frame_rgb)

        # Mostrar información de las manos detectadas
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                handedness = results.multi_handedness[idx].classification[0].label
                print(f"Mano detectada: {handedness}")

                for i, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    print(f"Punto {i}: ({x}, {y})")

                # Dibujar las manos detectadas
                mp_drawing.draw_landmarks(
                    output_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2)
                )

        # Mostrar el resultado
        cv2.imshow("gaturroña", output_frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
