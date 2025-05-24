# 🧠 YOLOv8 ile Gerçek Zamanlı Nesne Tespiti

Bu proje, [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) modeli kullanılarak, gerçek zamanlı olarak belirli nesnelerin webcam üzerinden tespit edilmesini sağlar. Model, yalnızca belirli sınıfları (örneğin: book, laptop, person, chair vs.) algılar ve bu nesneleri kare içine alarak tanımlar.


---

## 📸 Tanınan Sınıflar

Bu uygulama yalnızca aşağıdaki nesneleri algılar:

- book  
- cell phone  
- backpack  
- bottle  
- laptop  
- person  
- zebra  
- keyboard  
- mouse  
- chair
- knife
- clock  


---

## 🚀 Özellikler

- YOLOv8 kullanarak gerçek zamanlı nesne tespiti  
- Sadece belirlenen sınıfların ekranda gösterimi  
- Webcam ile canlı görüntü işleme  
- Tespit edilen nesneler için etiket ve güven skoru  

## 🔧 Gereksinimler

Python 3.8 veya üzeri sürüm önerilir.

Gerekli kütüphaneleri yüklemek için:

bash
pip install ultralytics opencv-python


---

## 📦 YOLOv8 Modeli

Bu proje, Ultralytics tarafından eğitilen yolov8n.pt (nano versiyon) modelini kullanır. Bu model hafif olduğu için düşük güçlü sistemlerde de çalışabilir.

## ▶ Kullanım

Aşağıdaki komutu çalıştırarak uygulamayı başlatabilirsiniz:

bash
python detect.py

Not: detect.py dosyasında aşağıdaki kod olmalıdır:

bash
## 💻 Python Kodları

Aşağıdaki kod, YOLOv8 ile sadece belirli nesneleri tespit eder:

from ultralytics import YOLO
import cv2

# YOLOv8 modelini yükle (yolov8n.pt en hafif modeldir)
model = YOLO("yolov8n.pt")

# Sadece bu sınıfları göstermek istiyoruz
TARGET_CLASSES = ['book', 'cell phone', 'backpack', 'bottle', 'laptop', 'person', 'zebra', 'keyboard', 'mouse', 'chair']

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


---

## 📂 yolo-object-detection/
├── detect.py             # YOLOv8 ile nesne tespit kodu

└── yolov8n.pt            # (İsteğe bağlı) Model dosyası — otomatik indirilebilir

## 📌 Notlar
Model dosyası (yolov8n.pt) ilk çalıştırmada otomatik olarak indirilir.

Kameranızın çalıştığından ve erişilebilir olduğundan emin olun.

Daha güçlü modeller için yolov8s.pt, yolov8m.pt, yolov8l.pt gibi varyantlar kullanılabilir.


---

## 🔗 QR Kod ve GitHub Linki

📎 [GitHub Projesi Linki]  
📷 QR kod PDF poster üzerinde yer almalıdır.


---

## 👥 Katkı Yapanlar

- **Kerem Yakaner** - 2405902031 - Bilgisayar Teknolojileri/Yapay Zeka Operatörlüğü
- **Emir Aksu** - 2405902003 - Bilgisayar Teknolojileri/Yapay Zeka Operatörlüğü


---

## 🔄 İş Bölümü

- **Kerem Yakaner**: [Kodun düzenlenip uygulanması, README yazımı, çıktıların kontrolü]  
- **Emir Aksu**: [Kod tasarımı, poster yapımı, README yazımı, çıktıların kontrolü]

---
