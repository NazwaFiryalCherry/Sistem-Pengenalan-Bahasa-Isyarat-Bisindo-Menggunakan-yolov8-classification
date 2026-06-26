from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-cls.pt")

    model.train(
        data="dataset",
        epochs=30,
        imgsz=224,
        batch=32,
        device=0,
        workers=0,  
        name="angka_bisindo"
    )

if __name__ == "__main__":
    main()