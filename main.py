import cv2

# Load pretrained Haar cascade classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

scale = 1

# Start video capture (0 = webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()

print("Face Detection Started....")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = frame.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fx = 1 / scale

    # Resize grayscale image
    small_img = cv2.resize(gray, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_LINEAR)
    small_img = cv2.equalizeHist(small_img)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        small_img,
        scaleFactor=1.1,
        minNeighbors=2,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        aspect_ratio = w / h
        color = (255, 0, 0)

        if 0.75 < aspect_ratio < 1.3:
            center = (int((x + w * 0.5) * scale), int((y + h * 0.5) * scale))
            radius = int((w + h) * 0.25 * scale)
            cv2.circle(img, center, radius, color, 3)
        else:
            cv2.rectangle(
                img,
                (int(x * scale), int(y * scale)),
                (int((x + w - 1) * scale), int((y + h - 1) * scale)),
                color,
                3
            )

        # Eye detection inside face region
        face_roi = small_img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=2,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(30, 30)
        )

        for (ex, ey, ew, eh) in eyes:
            center = (
                int((x + ex + ew * 0.5) * scale),
                int((y + ey + eh * 0.5) * scale)
            )
            radius = int((ew + eh) * 0.25 * scale)
            cv2.circle(img, center, radius, color, 3)

    # Show output
    cv2.imshow("Face Detection", img)

    key = cv2.waitKey(10)
    if key == 27 or key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()