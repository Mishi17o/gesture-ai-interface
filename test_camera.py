import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera FOUND at index {i}")
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Camera test - index {i}", frame)
            else:
                print(f"Index {i} opened but no frame")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"No camera at index {i}")

print("Test finished. Close window with q.")