from ultralytics import YOLO

# Eğittiğin modeli yüklenir
model = YOLO("runs/detect/train/weights/best.pt")

# Modelin daha önce hiç görmediği görsellerle test yapılır
results = model.predict(
    source="data/test",  
    imgsz=640,
    conf=0.25,
    save=True   
)

print("Tahminler tamamlandı. Sonuçlar 'runs/detect/predict' klasöründe.")