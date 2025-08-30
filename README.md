# 🍲 Kannada Recipe Assistant

A multimodal recipe assistant that detects ingredients from your camera, finds matching Kannada recipes, and answers your cooking questions in Kannada using LLMs.

---

## 🚀 Features

* 📸 **Ingredient Detection**

  * Capture photo using webcam in Google Colab.
  * Detect multiple ingredients using **Roboflow API** or **YOLOv8/YOLOv10** models.

* 🌐 **Ingredient Mapping**

  * Maps detected English ingredient names → Kannada using a bilingual dictionary.

* 📖 **Recipe Matching**

  * Uses [Kannada Recipe Dataset](https://www.kaggle.com/datasets/prajwalkumbar/recipe-dataset-in-kannada).
  * Finds top 3 closest recipes based on detected ingredients.

* 🤖 **LLM Integration**

  * Provides contextual answers in Kannada (powered by **Gemini**).
  * Example: *“ಈ ರೆಸಿಪಿಯಲ್ಲಿ ಮುಖ್ಯ ಪದಾರ್ಥ ಯಾವುದು?”*

* 🗣️ **Planned**: Speech-to-Text (STT) + Text-to-Speech (TTS) for full voice assistant experience.

---

## 📂 Project Workflow

1. **Capture Image**

   ```python
   from IPython.display import Javascript
   from google.colab import output
   # Capture and save photo as photo.jpeg
   ```

2. **Ingredient Detection**

   * **Model 1**: Roboflow API (`food-ingredients-dataset/3`)
   * **Model 2**: YOLOv10 (pretrained + custom training)
   * **Model 3**: Self-trained custom dataset

3. **Map Ingredients (English → Kannada)**

   ```python
   detected_ingredients_kannada = [
       english_to_kannada.get(ingredient, ingredient)
       for ingredient in detected_ingredients_english
   ]
   ```

4. **Find Recipes**

   ```python
   top_recipes = find_matching_recipes(detected_ingredients_kannada, df)
   ```

5. **LLM Response**

   ```python
   llm_response = get_kannada_response(user_question, context)
   ```

---

## 📊 Example Run

**Detected Ingredient**:

```text
['Green Lentils']
```

**Top 3 Matching Recipes**:

* ✅ Recipe 1 → Name + Ingredients + Procedure
* ✅ Recipe 2 → Name + Ingredients + Procedure
* ✅ Recipe 3 → Name + Ingredients + Procedure

**User Query**:

```text
ಈ ರೆಸಿಪಿಯಲ್ಲಿ ಮುಖ್ಯ ಪದಾರ್ಥ ಯಾವುದು?
```

**LLM Response**:

```text
ಈ ರೆಸಿಪಿಯಲ್ಲಿ ಮುಖ್ಯ ಪದಾರ್ಥವನ್ನು ಸ್ಪಷ್ಟವಾಗಿ ಉಲ್ಲೇಖಿಸಿಲ್ಲ. ಆದರೆ, ಸಾಮಾನ್ಯವಾಗಿ "ಲೋಫ್" ಎಂದರೆ ಮಾಂಸ ಅಥವಾ ಬೇಳೆ ಆಧಾರಿತವಾಗಿರುತ್ತದೆ...
```

---

## 🔧 Tech Stack

* **Python**, **Google Colab**
* **Computer Vision** → Roboflow, YOLOv8/YOLOv10
* **Dataset** → Kannada Recipes (Kaggle)
* **LLM** → Google Gemini API
* **Speech** → Planned (STT + TTS)

---

## 📌 Next Steps

* [ ] Add **speech-to-text** for user queries.
* [ ] Integrate **text-to-speech** to read out answers.
* [ ] Improve multi-ingredient detection accuracy.
* [ ] Deploy as a web app or mobile assistant.

---

✨ With this assistant, you can **point your camera at ingredients, get recipes in Kannada, and ask cooking questions naturally!**
