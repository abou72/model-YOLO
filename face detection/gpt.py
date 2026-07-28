import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
face_model = YOLO("yolov8n-face.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la caméra.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Détection des personnes
    result = model(frame, verbose=False)

    # Détection des visages
    face_result = face_model(frame, verbose=False)

    # Dessiner les personnes
    frame_result = result[0].plot()

    # Dessiner les visages sans les points clés
    frame_result = face_result[0].plot(
        img=frame_result,
        kpt_radius=0
    )

    cv2.imshow("Detection en temps reel", frame_result)

    # Appuyer sur X pour quitter
    if cv2.waitKey(1) & 0xFF == ord("x"):
        break

cap.release()
cv2.destroyAllWindows()