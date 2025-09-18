#%%
from ultralytics.data.utils import check_det_dataset
check_det_dataset("./SmartFridgle-1/data.yaml")
#%%
from ultralytics import YOLO

model = YOLO("yolo12n.yaml")
results = model.train(
    data="SmartFridgle-1/data.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    device=0,
    workers=0,
    cache=True
)
#%%
model = YOLO("C://users/kruth/runs/detect/train29/weights/best.pt")

#%%
from ultralytics import RTDETR

model1 = RTDETR("rtdetr-l.pt")
results = model1.train(
    data="SmartFridgle-1/data.yaml",
    epochs=25,
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    cache=True
)
