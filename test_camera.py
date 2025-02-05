import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not detected! Trying a different camera index...")
    cap = cv2.VideoCapture(1)  # Try second camera if the first one fails

if cap.isOpened():
    print("✅ Camera opened successfully!")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture a frame.")
            break

        cv2.imshow("Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    print("❌ Still unable to access the camera. Check your settings.")

cap.release()
