import cv2

stream = cv2.VideoCapture(0)

if not stream.isOpened():
    print("No Stream")
    exit()

while(True):
    ret, frame = stream.read()
    frame = cv2.flip(frame,1)
    if not ret:
        print("No More Stream")
        break

    cv2.imshow("Pong", frame)
    if cv2.waitKey(1) == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()