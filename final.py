#%%
import matplotlib.pyplot as plt
import PIL
#%%

import cv2
import matplotlib.pyplot as plt

# Step 1: Open webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Step 2: Capture one frame
ret, frame = cap.read()
cap.release()

if not ret:
    raise IOError("Failed to capture image")

# Step 3: Save the captured photo
photo_path = "photo.jpeg"
cv2.imwrite(photo_path, frame)

# Step 4: Display the captured image
# (convert BGR -> RGB for matplotlib)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

plt.imshow(frame_rgb)
plt.axis("off")
plt.title("Captured Image")
plt.show()
#%%
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import Counter

# Load your trained YOLO model
model = YOLO(r"C:/users/kruth/runs/detect/train29/weights/best.pt")

#%%
# Use the photo you captured locally (make sure photo.jpeg exists)
img_path = "photo.jpeg"

# Run prediction
results = model.predict(img_path, imgsz=640)

# Plot results
plt.imshow(results[0].plot()[:, :, ::-1])  # Convert BGR->RGB for matplotlib
plt.axis("off")
plt.title("Detected Ingredients")
plt.show()

#%% Extract detected classes safely
import numpy as np

detected_objects = []

# Make sure boxes.cls exists and is not None
cls_attr = getattr(results[0].boxes, "cls", None)
if cls_attr is not None:
    # If it's a PyTorch tensor, convert to numpy
    if hasattr(cls_attr, "detach"):
        detected_objects = cls_attr.detach().cpu().numpy()
    else:
        # Already numpy array
        detected_objects = np.array(cls_attr)

# Map to class names
detected_names = [model.names[int(cls)] for cls in detected_objects]

# Count occurrences
from collections import Counter
counts = Counter(detected_names)
print("🔎 Detected Ingredients:", counts)
#%%
english_to_kannada = {
    "Akabare Khursani": "ಅಕಬರೆ ಖುರ್ಸಾನಿ",
    "Apple": "ಸೇಬು",
    "Artichoke": "ಆರ್ಟಿಚೋಕ್",
    "Ash Gourd (Kubhindo)": "ಬೂದು ಸೌತೆಕಾಯಿ (ಕುಭಿಂಡೋ)",
    "Asparagus (Kurilo)": "ಅಸ್ಪಾರಾಗಸ್ (ಕುರಿಲೋ)",
    "Avocado": "ಅವಕಾಡೋ",
    "Bacon": "ಬೇಕನ್",
    "Bamboo Shoots (Tama)": "ಬಿದಿರು ಮೊಳೆಗಳು (ತಾಮಾ)",
    "Banana": "ಬಾಳೆಹಣ್ಣು",
    "Beans": "ಹುರಳಿಕಾಯಿ",
    "Beaten Rice (Chiura)": "ಅವಲಕ್ಕಿ (ಚಿಯುರಾ)",
    "Beef": "ಗೋಮಾಂಸ",
    "Beetroot": "ಬೆಟ್ರೂಟ್",
    "bell_pepper": "ದೋಣಮೆಣಸಿನಕಾಯಿ",
    "Bethu ko Saag": "ಬೆತು ಸೊಪ್ಪು",
    "Bitter Gourd": "ಹಾಗಲಕಾಯಿ",
    "Black beans": "ಕಪ್ಪು ಬೀನ್ಸ್",
    "Black Lentils": "ಕಪ್ಪು ತೊಗರಿ ಬೇಳೆ",
    "Bottle Gourd (Lauka)": "ಸೌತೆಕಾಯಿ (ಲೌಕಾ)",
    "Brinjal": "ಬದನೆಕಾಯಿ",
    "Broad Beans (Bakullo)": "ಹುರಳಿಕಾಯಿ (ಬಕುಲ್ಲೋ)",
    "Broccoli": "ಬ್ರೋಕೊಲಿ",
    "Buff Meat": "ಮಂಸ",
    "Butter": "ಬೆಣ್ಣೆ",
    "Cabbage": "ಎಲೆಕೋಸು",
    "Capsicum": "ದೋಣಮೆಣಸಿನಕಾಯಿ",
    "Carrot": "ಗಾಜರ",
    "Cassava (Ghar Tarul)": "ಕಸಾವಾ (ಘರ್ ತರುಲ್)",
    "Cauliflower": "ಹೂಕೋಸು",
    "Chayote (Iskus)": "ಚಯೋಟೆ (ಇಸ್ಕುಸ್)",
    "Cheese": "ಚೀಸ್",
    "Chicken": "ಕೋಳಿ ಮಾಂಸ",
    "Chicken Gizzards": "ಕೋಳಿ ಕಲ್ಲು",
    "Chickpeas": "ಕಡಲೆಕಾಯಿ",
    "Chili Pepper (Khursani)": "ಮೆಣಸಿನಕಾಯಿ (ಖುರ್ಸಾನಿ)",
    "Chili Powder": "ಮೆಣಸು ಪುಡಿ",
    "Chowmein Noodles": "ಚೌಮಿನ್ ನೂಡಲ್ಸ್",
    "Cinnamon": "ದಾಲ್ಚಿನ್ನಿ",
    "Coriander (Dhaniya)": "ಕೊತ್ತಂಬರಿ",
    "Corn": "ಜೋಳ",
    "Cornflakec": "ಕಾರ್ನ್ ಫ್ಲೇಕ್ಸ್",
    "Crab Meat": "ನೆಕ್ಕಿನ ಮಾಂಸ",
    "Cucumber": "ಸೌತೆಕಾಯಿ",
    "Egg": "ಮೊಟ್ಟೆ",
    "Farsi ko Munta": "ಫಾರ್ಸಿ ಸೊಪ್ಪು",
    "Fiddlehead Ferns (Niguro)": "ನಿಗುರೋ",
    "Fish": "ಮೀನು",
    "Garden cress (Chamsur ko saag)": "ಚಾಮ್ಸುರ್ ಸೊಪ್ಪು",
    "Garden Peas": "ಹಸಿರು ಬಟಾಣಿ",
    "Garlic": "ಬೆಳ್ಳುಳ್ಳಿ",
    "Ginger": "ಶುಂಠಿ",
    "Green Brinjal": "ಹಸಿರು ಬದನೆಕಾಯಿ",
    "Green Lentils": "ಹಸಿರು ತೊಗರಿ ಬೇಳೆ",
    "Green Mint (Pudina)": "ಪುದೀನಾ",
    "Green Peas": "ಹಸಿರು ಬಟಾಣಿ",
    "Green Soyabean (Hariyo Bhatmas)": "ಹಸಿರು ಸೋಯಾಬೀನ್",
    "Gundruk": "ಗುಂದ್ರುಕ್",
    "Ham": "ಹ್ಯಾಮ್",
    "Ice": "ಹಿಮ",
    "Jack Fruit": "ಹಲಸಿನಹಣ್ಣು",
    "Ketchup": "ಕೆಚಪ್",
    "Kimchi": "ಕಿಮ್ಚಿ",
    "Lapsi (Nepali Hog Plum)": "ಲಾಪ್ಸಿ",
    "Lemon (Nimbu)": "ನಿಂಬೆಹಣ್ಣು",
    "Lime (Kagati)": "ಲೈಮ್",
    "Long Beans (Bodi)": "ಉದ್ದ ಬೀನ್ಸ್",
    "Masyaura": "ಮಾಸ್ಯೌರಾ",
    "Mayonnaise": "ಮೇಯೋನೈಸ್",
    "Milk": "ಹಾಲು",
    "Minced Meat": "ಕಿಮಾ",
    "Moringa Leaves (Sajyun ko Munta)": "ನುಗ್ಗೆ ಸೊಪ್ಪು",
    "Mushroom": "ಅಣಬೆ",
    "Mutton": "ಮೆಕ್ಕೆಜೋಳ",
    "Noodle": "ನೂಡಲ್ಸ್",
    "Nutrela (Soya Chunks)": "ನುಟ್ರೆಲಾ (ಸೋಯಾ ಚಂಕ್ಸ್)",
    "Okra (Bhindi)": "ಬೆಂಡೆಕಾಯಿ",
    "Olive Oil": "ಆಲಿವ್ ಎಣ್ಣೆ",
    "Onion": "ಈರುಳ್ಳಿ",
    "Onion Leaves": "ಈರುಳ್ಳಿ ಎಲೆಗಳು",
    "orange": "ಕಿತ್ತಳೆ",
    "Palak (Indian Spinach)": "ಪಾಲಕ್",
    "Palungo (Nepali Spinach)": "ಪಾಲುಂಗೋ",
    "Paneer": "ಪನೀರ್",
    "Papaya": "ಪಪ್ಪಾಯಿ",
    "Pea": "ಬಟಾಣಿ",
    "Pear": "ಪಿಯರ್",
    "Pointed Gourd (Chuche Karela)": "ಚುಚೆ ಕರೆಲಾ",
    "Pork": "ಹಂದಿ ಮಾಂಸ",
    "Potato": "ಆಲೂಗಡ್ಡೆ",
    "Pumpkin (Farsi)": "ಕುಂಬಳಕಾಯಿ",
    "Radish": "ಮುಲಂಗು",
    "Rahar ko Daal": "ರಹರ್ ದಾಲ್",
    "Rayo ko Saag": "ರಾಯೋ ಸೊಪ್ಪು",
    "Red Beans": "ಕೆಂಪು ಬೀನ್ಸ್",
    "Red Lentils": "ಕೆಂಪು ತೊಗರಿ ಬೇಳೆ",
    "Rice (Chamal)": "ಅಕ್ಕಿ",
    "Sajjyun (Moringa Drumsticks)": "ನುಗ್ಗೆಕಾಯಿ",
    "Salt": "ಉಪ್ಪು",
    "Sausage": "ಸಾಸೇಜ್",
    "Seaweed": "ಸಮುದ್ರದ ಹುಲ್ಲು",
    "Snake Gourd (Chichindo)": "ಪಡವಲಕಾಯಿ",
    "Soy Sauce": "ಸೋಯಾ ಸಾಸ್",
    "Soyabean (Bhatmas)": "ಸೋಯಾಬೀನ್",
    "Sponge Gourd (Ghiraula)": "ಸ್ಫಾಂಜ್ ಗೋರ್ಡ್",
    "Stinging Nettle": "ಸಿಸ್ನು",
    "Strawberry": "ಸ್ಟ್ರಾಬೆರಿ",
    "Sugar": "ಸಕ್ಕರೆ",
    "Sweet Potato (Suthuni)": "ಶಿಪ್ಪುಗಡ್ಡೆ",
    "Taro Leaves (Karkalo)": "ಕರ್ಕಲ",
    "plasticsaveholder": "ಪ್ಲಾಸ್ಟಿಕ್ ಸೇವ್ ಹೋಲ್ಡರ್" # Placeholder for non-food item
}
#%% Safe extraction of detected English class names
import numpy as np

detected_ingredients_english = []

cls_attr = getattr(results[0].boxes, "cls", None)
if cls_attr is not None:
    # If it's a PyTorch tensor, convert to numpy
    if hasattr(cls_attr, "detach"):
        detected_objects = cls_attr.detach().cpu().numpy()
    else:
        # Already numpy array
        detected_objects = np.array(cls_attr)
    
    # Map indices to English names
    detected_ingredients_english = [model.names[int(cls)] for cls in detected_objects]

# Map them to Kannada using your dictionary
detected_ingredients_kannada = [
    english_to_kannada.get(ingredient, ingredient)
    for ingredient in detected_ingredients_english
]

print("🔎 Detected ingredients (English):", detected_ingredients_english)
print("🌿 Detected ingredients (Kannada):", detected_ingredients_kannada)
#%%
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Load the recipe dataset CSV directly into pandas
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "prajwalkumbar/recipe-dataset-in-kannada",
    "recipe_ingredients_and_procedure_kannada.csv"
)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
#%%
# # ✅ Extract ingredients properly from Roboflow result
# detected_ingredients_english = [
#     item["class"] for item in result["predictions"] if "class" in item
# ]

# # ✅ Map to Kannada
# detected_ingredients_kannada = [
#     english_to_kannada.get(ingredient, ingredient)
#     for ingredient in detected_ingredients_english
# ]

print("🔎 Detected ingredients (Kannada):", detected_ingredients_kannada)


# --- Function to find matching recipes ---
def find_matching_recipes(detected_ingredients, recipes_df):
    matching_scores = []

    for index, row in recipes_df.iterrows():
        recipe_ingredients = str(row["ingredients"])
        score = sum(ing in recipe_ingredients for ing in detected_ingredients)
        matching_scores.append((score, index))

    # Sort by score (high → low)
    matching_scores.sort(reverse=True, key=lambda x: x[0])

    # Pick top 3 with score > 0
    top_indices = [idx for score, idx in matching_scores if score > 0][:3]
    return recipes_df.iloc[top_indices]


# ✅ Get top 3 matches
top_matching_recipes = find_matching_recipes(detected_ingredients_kannada, df)

print("\n🍲 Top 3 Matching Recipes:")
# Instead of display(), use print or to_string() for VS Code console
print(top_matching_recipes[["recipe_name", "ingredients", "procedure"]].to_string(index=False))
#%%
context = top_matching_recipes.iloc[0]['procedure']
print(context)
#%%
# set gemini_api_key here
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now get the key from environment
api_key = os.getenv("GEMINI_API_KEY")

print("GEMINI_API_KEY loaded:", bool(api_key))  
#%%
from google import genai
from google.genai import types

client = genai.Client()  # now it can pick it up

def get_kannada_response(user_text, context, detected_ingredients_kannada):
    """
    Sends user speech (transcribed to text) and recipe context
    to Gemini model for generating a Kannada response.
    """
    prompt = f"""
    ನೀವು ಒಂದು ಕನ್ನಡ ಪಾಕಶಾಸ್ತ್ರ ಸಹಾಯಕರು.
    ನಿಮ್ಮಲ್ಲಿರುವ ಪದಾರ್ಥಗಳು: {', '.join(detected_ingredients_kannada)}.
    ಕೆಳಗಿನವು ನೀವು ಹುಡುಕಿದ ಪಾಕವಿಧಾನಗಳಾಗಿವೆ:

    {context}

    ಬಳಕೆದಾರರ ಪ್ರಶ್ನೆ:
    {user_text}

    ದಯವಿಟ್ಟು ಕನ್ನಡದಲ್ಲಿ ಸ್ಪಷ್ಟವಾದ ಉತ್ತರವನ್ನು ನೀಡಿ.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"
#%%
# Example usage (you can replace this with a way to get user input)
user_question = "ಈ ರೆಸಿಪಿಯಲ್ಲಿ ಮುಖ್ಯ ಪದಾರ್ಥ ಯಾವುದು?" # Example question in Kannada
llm_response = get_kannada_response(user_question, context, detected_ingredients_kannada)
print(llm_response)
#%%
import os
import gradio as gr
from PIL import Image, ImageDraw
#%%
# ---- Setup recipe context for demonstration ----
# These variables should be assigned from your previous pipeline!
image_path = "photo.jpeg"  # Adjust path as needed!
context = top_matching_recipes.iloc[0]['procedure']  # or any matched recipe procedure
detected_ingredients_kannada = ['ಬೆಳ್ಳುಳ್ಳಿ', 'ಪಟಂಟೋ', 'ಅಕ್ಕಿ']  # Example, replace with output from your detector
#%%
# ---- Ensure image exists ----
if not os.path.exists(image_path):
    img = Image.new("RGB", (600, 250), color=(240, 240, 240))
    d = ImageDraw.Draw(img)
    d.text((20, 100), "Place your image at photo.jpeg", fill=(20, 20, 20))
    img.save(image_path)
#%%
# --- Respond function for Gradio chatbot ---
def respond(user_message, chat_history):
    # Add user message to temporary conversation
    chat_history.append((user_message, ""))  # (user, placeholder)
    # Call LLM for reply
    reply = get_kannada_response(user_message, context, detected_ingredients_kannada)
    chat_history[-1] = (user_message, reply)  # Fill in with assistant reply
    return chat_history, chat_history
#%%
# ---- Gradio UI ----
with gr.Blocks(title="Kannada Recipe Assistant (RAG + Gemini)") as demo:
    gr.Markdown("### ಕನ್ನಡ ಪಾಕಶಾಸ್ತ್ರ ಸಹಾಯಕ — RAG + Gemini Chatbot")

    with gr.Row():
        # Left column: Show image
        with gr.Column(scale=1):
            img_display = gr.Image(
                value=image_path,
                label="Recipe photo (photo.jpeg)",
                interactive=False,
                type="pil"
            )
        # Right column: Chatbot
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="ಸಹಾಯಕ", type="messages")
            user_input = gr.Textbox(
                label="ನಿಮ್ಮ ಪ್ರಶ್ನೆ (ಕನ್ನಡ):",
                placeholder="ಉದಾ: ಈ ರೆಸಿಪಿಯಲ್ಲಿ ಮುಖ್ಯ ಪದಾರ್ಥ ಯಾವುದು?"
            )
            state = gr.State(value=[])  # chat history: list of pairs
            submit_btn = gr.Button("ಹೇಳಿ")

    def submit_click(user_text, chat_state):
        if not user_text or user_text.strip() == "":
            return gr.update(), chat_state  # No action if empty input
        pairs, new_state = respond(user_text, chat_state or [])
        return pairs, new_state

    submit_btn.click(fn=submit_click, inputs=[user_input, state], outputs=[chatbot, state])
    user_input.submit(fn=submit_click, inputs=[user_input, state], outputs=[chatbot, state])

    gr.Markdown(
        "**Notes:** Results are demo ONLY. All answers are generated by Gemini based on the retrieved recipe context and detected ingredients."
    )

demo.launch(share=True, prevent_thread_lock=True)
# %%
demo.close()
# %%
