# Kannada Recipe Assistant (RAG + Gemini)

An end-to-end intelligent assistant that detects ingredients from images (using YOLO), retrieves matching Kannada recipes via RAG, and provides conversational cooking support using — all wrapped in a clean web interface.

---

## 🌟 Features

- **Ingredient Detection:** Detects food ingredients from photos using / models.
- **Language Mapping:** Automatically maps detected ingredient names to Kannada.
- **Recipe Retrieval (RAG):** Retrieves relevant Kannada recipes from a large dataset using Retrieval-Augmented Generation.
- **Conversational Cooking AI:** Answers user questions in Kannada using , contextualized to selected recipes and detected ingredients.
- **Interactive UI:** Simple and intuitive interface built with for a smooth user experience.

---

## ⚙️ Example Workflow

1. Upload or capture a photo of ingredients.
2. The system detects all visible ingredients using YOLO.
3. Ingredient names are translated to Kannada.
4. Relevant Kannada recipes are retrieved using RAG.
5. Chat with the assistant in Kannada for cooking instructions, clarifications, or substitutions.

---

## 🛠️ Installation

### Prerequisites

- &#x20;3.9 or later
- CUDA-capable GPU (optional but recommended for faster YOLO inference)

### Clone and Install

```bash
git clone https://github.com/your-username/kannada-recipe-assistant.git
cd kannada-recipe-assistant
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
ROBOFLOW_API_KEY=your_roboflow_api_key_here
```

> ⚠️ **Never commit `.env` to version control** — it contains secret keys.

---

## 🚀 Usage

Run the app from your terminal:

```bash
python app.py
```

**Default behavior:**

- Captures image from webcam or uses `photo.jpeg`.
- Runs YOLO detection, translates ingredient names, retrieves best-matching Kannada recipes.
- Enables conversational cooking support in Kannada within the web interface.

---

## 🧠 Architecture Overview

```
Image → YOLO Detection → Ingredient Names → Kannada Mapping
      → RAG Search (Recipe Dataset) → Top Recipes
      → Conversational Q&A via Gemini → Gradio UI
```

---

## 📁 Project Structure

```
.
├── app.py                    # Main entry point
├── requirements.txt          # Dependencies
├── english_to_kannada.py     # Utility: English → Kannada mapping
├── data/                     # YOLO models + recipe datasets
├── .env                      # API keys (not tracked in Git)
└── README.md
```

---

## ⚡ Customization

- **Recipes:** Replace or extend the recipe dataset (CSV or other formats).
- **YOLO Model:** Train YOLO on additional or custom food categories.
- **Interface:** Modify the Gradio layout in `app.py` to match your branding or add new features.

---

## 🧪 Demo (Optional)

If you want, add screenshots or a GIF showing:

- Ingredient detection on an image
- Retrieved Kannada recipe
- Conversational support answering a cooking question

---

## 🤝 Contributing

1. Fork this repository
2. Create your feature branch: `git checkout -b feature/awesome-idea`
3. Commit your changes: `git commit -m 'Add awesome idea'`
4. Push to the branch: `git push origin feature/awesome-idea`
5. Open a Pull Request

---

## 🔒 Security Notes

- Do not expose or commit API keys.
- Use `.gitignore` to exclude `.env` and any sensitive datasets.
- Rotate your keys periodically if this project is public.

---

## 🙏 Acknowledgements

- &#x20;for object detection
- &#x20;for dataset management
- &#x20;for conversational AI
- &#x20;for the interactive app UI
- &#x20;for open recipe datasets

---
