#%%
import yaml
import os

# Define the data for the data.yaml file
data_yaml_content = {
    'train': "C:/Users/kruth/Downloads/model_eval/merged_dataset/train/images",
    'val': "C:/Users/kruth/Downloads/model_eval/merged_dataset/valid/images",
    'test': "C:/Users/kruth/Downloads/model_eval/merged_dataset/test/images",
    'nc': 103,
    'names': [
        "Akabare Khursani", "Apple", "Artichoke", "Ash Gourd (Kubhindo)",
        "Asparagus (Kurilo)", "Avocado", "Bacon", "Bamboo Shoots (Tama)",
        "Banana", "Beans", "Beaten Rice (Chiura)", "Beef", "Beetroot",
        "Bethu ko Saag", "Bitter Gourd", "Black Beans", "Black Lentils",
        "Bottle Gourd (Lauka)", "Brinjal", "Broad Beans (Bakullo)",
        "Broccoli", "Buff Meat", "Butter", "Cabbage", "Capsicum", "Carrot",
        "Cassava (Ghar Tarul)", "Cauliflower", "Chayote (Iskus)", "Cheese",
        "Chicken", "Chicken Gizzards", "Chickpeas", "Chili Pepper (Khursani)",
        "Chowmein Noodles", "Coriander (Dhaniya)", "Corn", "Cornflakes",
        "Crab Meat", "Cucumber", "Egg", "Farsi ko Munta", "Fiddlehead Ferns (Niguro)",
        "Fish", "Garden Cress (Chamsur ko Saag)", "Garden Peas", "Garlic",
        "Ginger", "Green Brinjal", "Green Lentils", "Green Mint (Pudina)",
        "Green Soyabean (Hariyo Bhatmas)", "Gundruk", "Ham", "Ice",
        "Jack Fruit", "Ketchup", "Kimchi", "Lapsi (Nepali Hog Plum)",
        "Lemon (Nimbu)", "Lime (Kagati)", "Long Beans (Bodi)", "Masyaura",
        "Mayonnaise", "Milk", "Minced Meat", "Moringa Leaves (Sajyun ko Munta)",
        "Mushroom", "Mutton", "Noodle", "Nutrela (Soya Chunks)", "Okra (Bhindi)",
        "Olive Oil", "Onion", "Onion Leaves", "Orange", "Palak (Indian Spinach)",
        "Palungo (Nepali Spinach)", "Paneer", "Papaya", "Pea", "Pear",
        "Pointed Gourd (Chuche Karela)", "Pork", "Potato", "Pumpkin (Farsi)",
        "Radish", "Rahar ko Daal", "Rayo ko Saag", "Red Beans", "Red Lentils",
        "Rice (Chamal)", "Sajjyun (Moringa Drumsticks)", "Sausage", "Seaweed",
        "Snake Gourd (Chichindo)", "Soy Sauce", "Soyabean (Bhatmas)", "Sponge Gourd (Ghiraula)",
        "Stinging Nettle", "Strawberry", "Sweet Potato (Suthuni)", "Taro Leaves (Karkalo)"
    ]
}

# Define the path to save the data.yaml file
merged_dataset_path = "merged_dataset"
data_yaml_path = os.path.join(merged_dataset_path, "data.yaml")

# Write the data to the data.yaml file
with open(data_yaml_path, "w") as f:
    yaml.dump(data_yaml_content, f)

print(f"✅ Created data.yaml file at: {data_yaml_path}")
#%%
from roboflow import Roboflow
import ultralytics
import os
from dotenv import load_dotenv

# Initialize Roboflow

# Load environment variables from .env file
load_dotenv()

# Get the key from environment
API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Use it to initialize Roboflow
rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project("food-ingredients-dataset-jxtbj")

print(ultralytics.__version__)
#%%
from ultralytics import YOLO

# This will automatically download YOLOv12-nano pretrained weights
model = YOLO("yolo12n.yaml")  # nano version

# Train on merged dataset
results = model.train(
    data='SmartFridgle-1/data.yaml',
    epochs=100,          # adjust to 100-120
    imgsz=640,           # image size
    batch=16,            # adjust depending on GPU memory
    device='0',         # use '0' for GPU, 'cpu' for CPU
    save_period=1,    # save every epoch
    amp=True     # automatic mixed precision
)
#%%
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Assume model is already trained, e.g., from previous step
# If not, load trained weights:
# model = YOLO("runs/detect/train/weights/best.pt")

# Evaluate model on validation set
metrics = model.val()
print(f"mAP@0.5: {metrics.metrics['mAP_0.5']:.4f}")
print(f"Precision: {metrics.metrics['precision']:.4f}")
print(f"Recall: {metrics.metrics['recall']:.4f}")