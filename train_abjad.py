from ultralytics import YOLO

def main():
    # Model Classification
    model = YOLO("yolov8n-cls.pt")

    # Training
    model.train(
        data="dataset",
        epochs=30,
        imgsz=224,
        batch=32,
        device=0,      
        workers=0,     
        name="abjad_bisindo"
    )

if __name__ == "__main__":
    main()