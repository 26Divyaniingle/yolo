from ultralytics import YOLO


def main():
    # Load pretrained YOLOv8 Nano model
    model = YOLO("yolov8n.pt")

    # Train the model
    results = model.train(
        data="yolo_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        patience=10,
        optimizer="auto",
        lr0=0.01,
        pretrained=True,
        project="runs/detect",
        name="day5_yolov8n",
        exist_ok=True
    )

    print("\nTraining completed!")
    print("Training results saved in:")
    print("runs/detect/day5_yolov8n")


if __name__ == "__main__":
    main()