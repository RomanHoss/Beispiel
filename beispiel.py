from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1. YOLO-Modell laden (vortrainiertes Modell)
model = YOLO('yolov8n.pt')  # 'yolov8n.pt' ist das Nano-Modell (schnell und leicht)

# 2. Bild laden
image_path = 'C:/Users/Roman/Downloads/fu.jpg'  # Pfad zu deinem Bild
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konvertiere zu RGB für die Anzeige

# 3. Objekterkennung durchführen
results = model(image_rgb)

# 4. Ergebnisse anzeigen
# Ergebnisse werden direkt auf das Bild gezeichnet
annotated_image = results[0].plot()

# Zeige das Bild mit den erkannten Objekten
plt.imshow(annotated_image)
plt.axis('off')
plt.show()

# 5. Zusätzliche Informationen ausgeben
for result in results[0].boxes:
    print(f"Objekt: {result.cls}, Wahrscheinlichkeit: {result.conf:.2f}, Box: {result.xyxy}")
# test_yolo.py
from ultralytics import YOLO

def test_yolo_model():
    model = YOLO('yolov8n.pt')  # Lade das Modell
    assert model is not None, "Das Modell konnte nicht geladen werden."

def test_image_processing():
    image_path = 'example.jpg'  # Beispielbild
    image = cv2.imread(image_path)
    assert image is not None, "Das Bild konnte nicht geladen werden."

if __name__ == "__main__":
    test_yolo_model()
    test_image_processing()
    print("Alle Tests erfolgreich!")

