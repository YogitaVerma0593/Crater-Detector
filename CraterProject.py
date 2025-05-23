from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(data="C:/Users/hp/Craters Detector.v1i.yolov8/data.yaml", epochs=30)  

