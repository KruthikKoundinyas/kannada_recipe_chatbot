#%%
import gradio as gr
from PIL import Image
import requests
from io import BytesIO
#%%
# --- Model Map (example, update as needed for your YOLO/rtdetr) ---
MODEL_PATHS = {
    "yolov8n.pt": r"C:/models/yolov8n.pt",
    "yolo11n.pt": r"C:/models/yolo11n.pt",
    "rtdetr-l.pt": r"C:/models/rtdetr-l.pt",
    "best.pt": r"C:/users/kruth/runs/detect/train29/weights/best.pt"
}
#%%
# --- Your FOOD DETECTION + TRANSLATION ---
def load_image_from_url(url):
    try:
        r = requests.get(url)
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None
#%%
def process_image_and_model(image, webcam_image, image_url, model_name):
    try:
        pil_img = webcam_image if webcam_image else image
        if pil_img is None and image_url:
            pil_img = load_image_from_url(image_url)
        if pil_img is None:
            return "No valid image provided.", [], "", None, None

        from ultralytics import YOLO
        detected_ingredients_kannada = []
        detected_classes = []
        try:
            model_path = MODEL_PATHS[model_name]
            model = YOLO(model_path)
            print("Image to be saved:", pil_img)
            if pil_img:
                pil_img.save("temp_input.jpg")
            else:
                print("PIL image not saved.")

            results = model.predict("temp_input.jpg", imgsz=416)
            # The attribute access below is speculative; replace with your real inference!
            try:
                # For YOLO standard output
                cls_attr = getattr(results[0].boxes, "cls", None)
                import numpy as np
                if cls_attr is not None:
                    detected_objects = cls_attr.detach().cpu().numpy() if hasattr(cls_attr, "detach") else np.array(cls_attr)
                    detected_classes = [model.names[int(cls)] for cls in detected_objects]
            except Exception:
                # For custom output
                detected_classes = [result["class_label"] for result in results]
            
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
        
        except Exception as e:
            return f"Error in detection: {str(e)}", [], "", None, None

        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "prajwalkumbar/recipe-dataset-in-kannada",
            "recipe_ingredients_and_procedure_kannada.csv"
        )
        def normalize_text(text):
            return text.lower().strip()

        def find_matching_recipes(detected_ingredients, recipes_df):
            matching_scores = []
            for idx, row in recipes_df.iterrows():
                recipe_ingredients = normalize_text(str(row["ingredients"]))
                score = sum(normalize_text(ing) in recipe_ingredients for ing in detected_ingredients)
                matching_scores.append((score, idx))
            matching_scores.sort(reverse=True, key=lambda x: x[0])
            top_indices = [idx for score, idx in matching_scores if score > 0][:3]
            return recipes_df.iloc[top_indices] if top_indices else None
        
        detected_ingredients_kannada = [
            english_to_kannada.get(str(ingredient.capitalize()), str(ingredient))
            for ingredient in detected_classes
        ]
        top_matching_recipes = find_matching_recipes(detected_ingredients_kannada, df)
        if top_matching_recipes is not None and len(top_matching_recipes) > 0:
            context = top_matching_recipes.iloc[0]["procedure"]
        else:
            context = "No matching recipe found"

        status = f"✅ Detected: {', '.join(detected_ingredients_kannada) if detected_ingredients_kannada else 'No ingredients detected.'}\nModel used: {model_name}"

        return status, detected_ingredients_kannada, context, pil_img, top_matching_recipes

    except Exception as e:
        return f"Error in processing: {str(e)}", [], "", None, None
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
#%%# --- Respond function for Gradio chatbot ---
def respond(user_message, chat_history, context, detected_ingredients_kannada):
    reply = get_kannada_response(user_message, context, detected_ingredients_kannada)
    chat_history.append((user_message, reply))
    return chat_history, chat_history
#%%
with gr.Blocks() as demo:
    gr.Markdown("## Kannada Recipe Assistant (Image → Model → Chatbot)")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image", type="pil", interactive=True)
            webcam_input = gr.Image(label="Capture via Webcam", source="webcam", type="pil", interactive=True)
            image_url = gr.Textbox(label="Or Enter Image URL", placeholder="https://example.com/image.jpg")
            model_selector = gr.Dropdown(
                choices=list(MODEL_PATHS.keys()),
                label="Select Detection Model", 
                value="yolov8n.pt"
            )
            process_btn = gr.Button("Process Image")
            status = gr.Markdown("Awaiting image upload...")
        with gr.Column():
            preview_img = gr.Image(label="Preview")

    detected_ingredients_state = gr.State([])
    context_state = gr.State("")
    top_recipes_state = gr.State(None)

    # Chatbot components start hidden individually
    chatbot = gr.Chatbot(label="ಸಹಾಯಕ", visible=False)
    user_input = gr.Textbox(label="ನಿಮ್ಮ ಪ್ರಶ್ನೆ (ಕನ್ನಡ):", placeholder="ರೆಸಿಪಿಯಲ್ಲಿ ಮುಖ್ಯ ಪದಾರ್ಥ ಯಾವುದು?", visible=False)
    submit_btn = gr.Button("ಹೇಳಿ", visible=False)
    chat_history = gr.State([])

    def on_process_clicked(img, webcam_img, url, model):
        status_msg, detected_ingredients, context, preview, top_recipes = process_image_and_model(
            img, webcam_img, url, model
        )
        print("DEBUG in callback: context =", repr(context))
        visible = True if context and context != "No matching recipe found" else False
        return (
            status_msg, preview, detected_ingredients, context, top_recipes,
            gr.update(visible=visible),
            gr.update(visible=visible),
            gr.update(visible=visible)
        )

    process_btn.click(
        fn=on_process_clicked,
        inputs=[image_input, webcam_input, image_url, model_selector],
        outputs=[
            status, preview_img, detected_ingredients_state,
            context_state, top_recipes_state,
            chatbot, user_input, submit_btn
        ]
    )

    submit_btn.click(
        fn=respond,
        inputs=[user_input, chat_history, context_state, detected_ingredients_state],
        outputs=[chatbot, chat_history]
    )
    user_input.submit(
        fn=respond,
        inputs=[user_input, chat_history, context_state, detected_ingredients_state],
        outputs=[chatbot, chat_history]
    )
#%%
demo.launch(share=True, prevent_thread_lock=True) #https://simple-veganista.com/wp-content/uploads/2020/03/house-salad-recipe-ingredients-680x1014.jpg
#%% # Test locally
demo.launch(share=False, prevent_thread_lock=True) # https://th.bing.com/th/id/OIP.5okzrDqhBctzdQw4y4_2XQHaE7?w=239&h=180&c=7&r=0&o=7&dpr=2.1&pid=1.7&rm=3
#%%
demo.close()
# %%
