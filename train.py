import os
import re
from ultralytics import YOLO
import shutil
from ultralytics.utils import SETTINGS

# Etiketsiz görsel verilerine eğitim öncesi ön hazırlık kısmıdır
""" 
image_folder = 'data/images/train'
label_folder = 'data/labels/train'

# Eski etiket klasörünü temizle
if os.path.exists(label_folder):
    shutil.rmtree(label_folder)
os.makedirs(label_folder)

# Etiketsiz görsel verilerine txt dosyası oluşturur ve etiketlendirir. 
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg'):
        basename = os.path.splitext(filename)[0]

        # En sondaki "_0" veya "_1" değerini yakalar
        match = re.search(r'_(0|1)(?:\s*\(\d+\))?$', basename)
        if match:
            class_id = match.group(1)

            txt_path = os.path.join(label_folder, basename + '.txt')
            with open(txt_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
        else:
            print(f"Etiket bulunamadı: {filename}") 
"""


def main():
    model = YOLO("yolo11n.pt")
    # data.yaml dosyasının bilgisayarınızdaki tam yolunu buraya yazın
    results = model.train(data="Change the path according to your setup for data.yaml", epochs=100, imgsz=640)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    main() 
