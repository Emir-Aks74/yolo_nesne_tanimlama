from ultralytics import YOLO
import cv2

# YOLOv8 modelini yükle (yolov8n.pt en hafif modeldir)
model = YOLO("yolov8n.pt")

# Sadece bu sınıfları göstermek istiyoruz
TARGET_CLASSES = ['book', 'cell phone', 'backpack', 'bottle','person','laptop','keyboard','chair','zebra','knife','toothbrush','clock','skateboard','mouse']

# Kamera aç
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Model ile tahmin yap
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            # Sadece hedef sınıfları kontrol et
            if label in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow("YOLOv8 - Nesne Tespiti", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC çıkış
        break

cap.release()
cv2.destroyAllWindows()