#%%
import os
from dotenv import load_dotenv
from roboflow import Roboflow

# Load environment variables from .env file
load_dotenv()

# Get the key from environment
API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Use it to initialize Roboflow
rf = Roboflow(api_key=API_KEY)


# List of datasets to download
datasets_to_download = [
    {"workspace": "recipe-x2naf", "project": "food-ingredients-dataset-e9gbi", "version": 1, "format": "yolov8"},
    {"workspace": "recipe-x2naf", "project": "food-ingredients-7txer-noh2z", "version": 1, "format": "yolov8"},
    {"workspace": "recipe-x2naf", "project": "smartfridgle-aohhj", "version": 1, "format": "yolov8"},
    {"workspace": "recipe-x2naf", "project": "food-ingredients-dataset-jxtbj", "version": 3, "format": "yolov8"}
]

downloaded_datasets = {}

for dataset_info in datasets_to_download:
    try:
        project = rf.workspace(dataset_info["workspace"]).project(dataset_info["project"])
        version = project.version(dataset_info["version"])
        dataset = version.download(dataset_info["format"])

        # dataset.location is already the folder with train/valid/test
        downloaded_datasets[dataset_info["project"]] = dataset.location
        print(f"✅ Dataset '{dataset_info['project']}' version {dataset_info['version']} ready at: {dataset.location}")

    except Exception as e:
        print(f"❌ Error downloading dataset '{dataset_info['project']}' version {dataset_info['version']}: {e}")

#%%
import os
import shutil

# Define the list of allowed classes
allowed_classes_english = [
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

# Create a mapping from class name to class index (assuming YOLO format with index)
# You'll need to determine the class index for each allowed class from your dataset's data.yaml file
# For now, we'll use a placeholder. You'll need to replace this with the actual mapping.
class_name_to_index = {
    "Akabare Khursani": 0, "Apple": 1, "Artichoke": 2, "Ash Gourd (Kubhindo)": 3,
    "Asparagus (Kurilo)": 4, "Avocado": 5, "Bacon": 6, "Bamboo Shoots (Tama)": 7,
    "Banana": 8, "Beans": 9, "Beaten Rice (Chiura)": 10, "Beef": 11, "Beetroot": 12,
    "Bethu ko Saag": 13, "Bitter Gourd": 14, "Black Beans": 15, "Black Lentils": 16,
    "Bottle Gourd (Lauka)": 17, "Brinjal": 18, "Broad Beans (Bakullo)": 19,
    "Broccoli": 20, "Buff Meat": 21, "Butter": 22, "Cabbage": 23, "Capsicum": 24, "Carrot": 25,
    "Cassava (Ghar Tarul)": 26, "Cauliflower": 27, "Chayote (Iskus)": 28, "Cheese": 29,
    "Chicken": 30, "Chicken Gizzards": 31, "Chickpeas": 32, "Chili Pepper (Khursani)": 33,
    "Chowmein Noodles": 34, "Coriander (Dhaniya)": 35, "Corn": 36, "Cornflakes": 37,
    "Crab Meat": 38, "Cucumber": 39, "Egg": 40, "Farsi ko Munta": 41, "Fiddlehead Ferns (Niguro)": 42,
    "Fish": 43, "Garden Cress (Chamsur ko Saag)": 44, "Garden Peas": 45, "Garlic": 46,
    "Ginger": 47, "Green Brinjal": 48, "Green Lentils": 49, "Green Mint (Pudina)": 50,
    "Green Soyabean (Hariyo Bhatmas)": 51, "Gundruk": 52, "Ham": 53, "Ice": 54,
    "Jack Fruit": 55, "Ketchup": 56, "Kimchi": 57, "Lapsi (Nepali Hog Plum)": 58,
    "Lemon (Nimbu)": 59, "Lime (Kagati)": 60, "Long Beans (Bodi)": 61, "Masyaura": 62,
    "Mayonnaise": 63, "Milk": 64, "Minced Meat": 65, "Moringa Leaves (Sajyun ko Munta)": 66,
    "Mushroom": 67, "Mutton": 68, "Noodle": 69, "Nutrela (Soya Chunks)": 70, "Okra (Bhindi)": 71,
    "Olive Oil": 72, "Onion": 73, "Onion Leaves": 74, "Orange": 75, "Palak (Indian Spinach)": 76,
    "Palungo (Nepali Spinach)": 77, "Paneer": 78, "Papaya": 79, "Pea": 80, "Pear": 81,
    "Pointed Gourd (Chuche Karela)": 82, "Pork": 83, "Potato": 84, "Pumpkin (Farsi)": 85,
    "Radish": 86, "Rahar ko Daal": 87, "Rayo ko Saag": 88, "Red Beans": 89, "Red Lentils": 90,
    "Rice (Chamal)": 91, "Sajjyun (Moringa Drumsticks)": 92, "Sausage": 93, "Seaweed": 94,
    "Snake Gourd (Chichindo)": 95, "Soy Sauce": 96, "Soyabean (Bhatmas)": 97, "Sponge Gourd (Ghiraula)": 98,
    "Stinging Nettle": 99, "Strawberry": 100, "Sweet Potato (Suthuni)": 101, "Taro Leaves (Karkalo)": 102
}

# Get the list of allowed class indices
allowed_class_indices = [class_name_to_index[name] for name in allowed_classes_english if name in class_name_to_index]
#%%
downloaded_datasets = {
    "FOOD-INGREDIENTS-dataset-1": "FOOD-INGREDIENTS-dataset-1",
    "FOOD-INGREDIENTS-dataset-3": "FOOD-INGREDIENTS-dataset-3",
    "SmartFridgle-1": "SmartFridgle-1",
    "food-ingredients-1": "food-ingredients-1"
}

# Iterate through each downloaded dataset
import yaml

for project_name, dataset_path in downloaded_datasets.items():
    print(f"Processing dataset: {project_name} at {dataset_path}")

    # Load dataset's data.yaml
    yaml_path = os.path.join(dataset_path, "data.yaml")
    with open(yaml_path, "r") as f:
        data_config = yaml.safe_load(f)
    dataset_classes = data_config["names"]

    # Build mapping: dataset index → global index
    dataset_index_to_global_index = {
        i: class_name_to_index[name]
        for i, name in enumerate(dataset_classes)
        if name in class_name_to_index
    }

    # Define paths
    splits = ["train", "valid", "test"]
    for split in splits:
        labels_path = os.path.join(dataset_path, split, "labels")
        filtered_labels_path = os.path.join(dataset_path, split, "filtered_labels")
        os.makedirs(filtered_labels_path, exist_ok=True)

        if not os.path.exists(labels_path):
            print(f"Labels directory not found: {labels_path}. Skipping.")
            continue

        # Process labels
        for filename in os.listdir(labels_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(labels_path, filename)
                filtered_filepath = os.path.join(filtered_labels_path, filename)

                with open(filepath, "r") as infile, open(filtered_filepath, "w") as outfile:
                    for line in infile:
                        parts = line.split()
                        if len(parts) > 0:
                            class_index = int(parts[0])
                            if class_index in dataset_index_to_global_index:
                                global_index = dataset_index_to_global_index[class_index]
                                outfile.write(f"{global_index} " + " ".join(parts[1:]) + "\n")

                # Remove empty files
                if os.path.getsize(filtered_filepath) == 0:
                    os.remove(filtered_filepath)

    print(f"✅ Filtering + remapping complete for dataset: {project_name}")
#%%
import os

# Assuming your filtered labels are in 'filtered_labels' subdirectories as created in the previous cell
# and your images are in 'images' subdirectories within each dataset's train, valid, and test folders.
# Adjust paths if necessary.
downloaded_datasets = {
    "FOOD-INGREDIENTS-dataset-1": "/content/FOOD-INGREDIENTS-dataset-1",
    "FOOD-INGREDIENTS-dataset-3": "/content/FOOD-INGREDIENTS-dataset-3",
    "SmartFridgle-1": "/content/SmartFridgle-1",
    "food-ingredients-1": "/content/food-ingredients-1",
    "merged_dataset": "/content/merged_dataset"
}

# Iterate through each downloaded dataset
for project_name, dataset_path in downloaded_datasets.items():
    print(f"Checking dataset: {project_name} at {dataset_path}")

    # Define paths to train, validation, and test image and filtered label directories
    train_images_path = os.path.join(dataset_path, "train", "images")
    valid_images_path = os.path.join(dataset_path, "valid", "images")
    test_images_path = os.path.join(dataset_path, "test", "images")

    filtered_train_labels_path = os.path.join(dataset_path, "train", "filtered_labels")
    filtered_valid_labels_path = os.path.join(dataset_path, "valid", "filtered_labels")
    filtered_test_labels_path = os.path.join(dataset_path, "test", "filtered_labels")

    # Function to check for images without labels in a given directory pair
    def check_images_without_labels(images_dir, filtered_labels_dir):
        if not os.path.exists(images_dir):
            print(f"Images directory not found: {images_dir}. Skipping check.")
            return
        if not os.path.exists(filtered_labels_dir):
            print(f"Filtered labels directory not found: {filtered_labels_dir}. Skipping check.")
            return

        images_without_labels = []
        for filename in os.listdir(images_dir):
            # Assuming image files have extensions like .jpg, .jpeg, .png, etc.
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_name_without_ext = os.path.splitext(filename)[0]
                label_filename = image_name_without_ext + ".txt"
                label_filepath = os.path.join(filtered_labels_dir, label_filename)

                # Check if the corresponding filtered label file exists and is not empty
                if not os.path.exists(label_filepath) or os.path.getsize(label_filepath) == 0:
                    images_without_labels.append(filename)

        if images_without_labels:
            print(f"Images without labels found in {images_dir}:")
            for img_file in images_without_labels:
                print(f"- {img_file}")
        else:
            print(f"No images without labels found in {images_dir}.")

    # Check images without labels for train, validation, and test sets
    print("Checking train set...")
    check_images_without_labels(train_images_path, filtered_train_labels_path)
    print("Checking validation set...")
    check_images_without_labels(valid_images_path, filtered_valid_labels_path)
    print("Checking test set...")
    check_images_without_labels(test_images_path, filtered_test_labels_path)

    print(f"Check complete for dataset: {project_name}\n")
#%%
import shutil

merged_root = "/content/merged_dataset"
splits = ["train", "valid", "test"]

# Create merged folder structure
for split in splits:
    os.makedirs(os.path.join(merged_root, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(merged_root, split, "labels"), exist_ok=True)

# Copy from each dataset
for project_name, dataset_path in downloaded_datasets.items():
    for split in splits:
        images_dir = os.path.join(dataset_path, split, "images")
        labels_dir = os.path.join(dataset_path, split, "filtered_labels")

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            continue

        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                img_name = f"{project_name}_{img_file}"  # avoid collisions
                lbl_name = os.path.splitext(img_name)[0] + ".txt"

                # Copy image
                shutil.copy(
                    os.path.join(images_dir, img_file),
                    os.path.join(merged_root, split, "images", img_name)
                )

                # Copy label if exists, otherwise skip
                original_lbl = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")
                if os.path.exists(original_lbl):
                    shutil.copy(
                        original_lbl,
                        os.path.join(merged_root, split, "labels", lbl_name)
                    )
print("✅ Merging complete.")
#%%
import os

for dataset_path in downloaded_datasets.values():
    print(f"\nDataset at: {dataset_path}")
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files[:5]:  # show first 5 files per folder
            print(f"{subindent}{f}")
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
# from ultralytics import RTDETR

# model1 = RTDETR("rtdetr-l.pt")
# results = model1.train(
#     data="SmartFridgle-1/data.yaml",
#     epochs=25,
#     imgsz=640,
#     batch=8,
#     device=0,
#     workers=0,
#     cache=True
# )