import cv2
import mediapipe as mp

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Open the webcam stream
stream = cv2.VideoCapture(0) # uses second camera on device 

if not stream.isOpened():
    print("No Stream")
    exit()

while(True):
    ret, frame = stream.read()
    frame = cv2.flip(frame, 1) 

    if not ret:
        print("No More Stream")
        break

    # Convert the frame to RGB (MediaPipe requires RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    result = hands.process(rgb_frame)

    # If hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw the landmarks on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate bounding box for hand
            x_min, y_min = 1, 1
            x_max, y_max = 0, 0

            for landmark in hand_landmarks.landmark:
                x_min = min(x_min, landmark.x) # we can use these variables to move the pong paddles around the screen
                y_min = min(y_min, landmark.y)
                x_max = max(x_max, landmark.x)
                y_max = max(y_max, landmark.y)

            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = (x_min * w, y_min * h, x_max * w, y_max * h)

            # Draw a bounding box around the hand
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

    # Show the frame with the bounding boxes
    cv2.imshow("Pong", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all OpenCV windows
stream.release()
cv2.destroyAllWindows()