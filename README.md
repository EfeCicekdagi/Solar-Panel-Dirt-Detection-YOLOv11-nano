# 🌞 Solar Panel Dirt Detection YOLOv11 nano

This repository contains code to **train** and **test** a YOLOv11‑nano model for detecting **dirt vs. clean** solar panel states.

- Framework: **Ultralytics YOLO**
- Model: **yolo11n.pt** (COCO-pretrained)
- Dataset: *Solar Panels Dirt Detection* (Kaggle, by malkamahira, **CC0: Public Domain**)

---

## 📂 Repository Structure

- data
  - images
    - train  # 80% of images
    - val  # 20% of images
  - labels
    - train
    - val
  - test
  - data.yaml
- runs
- test.py
- train.py
- yolo11n.pt
  
---

## 🧱 Dataset & Labels

**Name:** Solar Panels Dirt Detection  
- **Source:** [Kaggle — Solar Panels Dirt Detection](https://www.kaggle.com/datasets/malkamahira/solar-panels-dirt-detection)  
- **License:** **Unknown** (No explicit license information provided on the dataset page at the time of download)

> Because the license is not specified, please check the dataset’s Kaggle page for any usage restrictions or attribution requirements before using it in commercial or public projects.
> In this project, the images have been manually split into **80% training** and **20% validation** sets.  
> Make sure your dataset matches the following structure before training

---
## ⚙️ Setup

```bash
# (Optional) Create a clean env
conda create -n yolo-solar python=3.10 -y
conda activate yolo-solar

# Install dependencies
pip install ultralytics opencv-python numpy pandas matplotlib

# (GPU) Ensure CUDA-compatible PyTorch is installed
python -c "import torch; print(torch.cuda.is_available())" 
```
---
## 🚀 Training

- To start training the YOLOv11-nano model on your dataset  
- Loads COCO-pretrained yolo11n.pt checkpoint.  
- Trains for 100 epochs at 640×640 resolution (adjustable).  
- Update the data.yaml path in train.py:  
  - results = model.train(data="C:/path/to/data.yaml", epochs=100, imgsz=640)
- Results are saved to:
  - weights/best.pt — best performing weights
  - weights/last.pt — last epoch weights
  - results.png — training curves (loss, mAP, precision, recall)


---

## 🔎 Inference / Testing

After training, you can run inference on **your own test images**:

> **Note:** Please prepare and place your own test images in the `data/test` directory  
> (This project does not include test images due to dataset license and size).

---

## 🧰 Tips & Troubleshooting

- Dataset Paths  
  - Ensure data.yaml points to the correct images/train, images/val, labels/train, and labels/val directories.

- Windows Multiprocessing train.py uses:
  - from multiprocessing import freeze_support
- GPU Acceleration:    
  - If GPU is not used, check PyTorch CUDA support  
- Auto-Labeling (Optional)  
  - train.py contains a commented-out block to auto-generate YOLO labels from filenames ending with _0 or _1.

---

## 📜 License & Credit

- **Code:** Licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.  
- **Dataset:** [Kaggle — Solar Panels Dirt Detection](https://www.kaggle.com/datasets/malkamahira/solar-panels-dirt-detection) — **Unknown License**  
  - No explicit license information provided on the dataset page at the time of download.  
  - Please check the dataset’s Kaggle page for any updates or restrictions before public/commercial use.
