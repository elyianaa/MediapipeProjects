import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
mp_face_mesh = mp.solutions.face_mesh


# get realtime webcam feed
cap = cv2.VideoCapture(0)

# initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()

        # recolor feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # make detections
        results = holistic.process(image)
        # print(results.face_landmarks)

        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        # recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )

        # right hand
        # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=1),
        #                           mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1)
        #                           )
        #
        # # left hand
        # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),
        #                           mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=1)
        #                           )
        #
        # # pose detections
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=1),
        #                           mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1)
        #                           )
        #
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

