# 🥘 Kannada Recipe Assistant

**YOLO + Gradio + Gemini**

An end-to-end intelligent assistant that detects ingredients from images (using ), retrieves matching Kannada recipes from a local dataset, and provides conversational cooking support in Kannada using  — all through a seamless Gradio web interface.

---

## 🌟 Features

* **Ingredient Detection:** Detects food ingredients from uploaded or captured images using YOLO models.
* **Language Mapping:** Automatically translates detected ingredient names to Kannada.
* **Recipe Retrieval:** Fetches relevant Kannada recipes from a local CSV dataset.
* **Conversational Cooking AI:** Answers user questions in Kannada using Gemini, with context from the selected recipe and detected ingredients.
* **Interactive UI:** Clean, responsive interface powered by Gradio.
* **Mobile-Friendly:** Supports image capture directly from your phone via link or upload.

---

## ⚙️ Example Workflow

1. Upload or capture a photo of ingredients using webcam, desktop, or phone.
2. The system detects all visible ingredients using YOLO.
3. Ingredient names are translated to Kannada.
4. Relevant Kannada recipes are retrieved from the local dataset.
5. Chat with the assistant in Kannada for step-by-step cooking help and clarifications.

---

## 🛠️ Installation

### Prerequisites

* Python3.9+
* NVIDIA CUDA-capable GPU (recommended for faster YOLO inference)

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

> ⚠️ **Never commit `.env` to version control** — it contains your secret API keys.

---

## 🚀 Usage

Run the app from your terminal:

```bash
python User_input.py
```

**What happens:**

* Captures image from webcam, phone (via link/upload), or uses `photo.jpeg`.
* Runs YOLO detection, translates ingredient names, and retrieves matching Kannada recipes.
* Enables conversational cooking support in Kannada within the Gradio web interface.

---

## 🧠 Architecture Overview

```
Image 
 → YOLO Detection 
 → Ingredient Names 
 → Kannada Mapping
 → Recipe Search (Local Dataset)
 → Top Kannada Recipes
 → Conversational Q&A via Gemini
 → Gradio UI
```

---

## 📁 Project Structure

```
.
├── .venv/                     # Python virtual environment
├── food-ingredients-1/        # Auxiliary ingredient data
├── FOOD-INGREDIENTS-1/
├── FOOD_INGREDIENTS-1/
├── merged_dataset/             # Dataset merges (if any)
├── SmartFridge-1/               # Related datasets/scripts
├── ultralytics/                 # YOLO-related code/models
├── .env                         # API keys (not tracked in Git)
├── .gitignore
├── best.pt                      # YOLO trained model (weights)
├── Chat_interface.py            # Chat interface scripts
├── DataSet_Preprocessing.py     # Dataset processing scripts
├── Model_training.py             # YOLO model training scripts
├── photo.jpeg                   # Example input image
├── README.md
├── requirements.txt
├── rtdetr-l.pt                  # Additional YOLO model weights
├── temp_input.jpg               # Latest processed image
├── test.py                      # Test scripts
├── User_input.py                 # Main entry point
├── yolov8n.pt                   # YOLO model weights
├── yolo11n.pt                   # YOLO model weights
```

---

## ⚡ Customization

* **Recipes:** Replace or extend the local recipe dataset CSV to add new recipes.
* **YOLO Model:** Train YOLO on additional or custom food categories using your own images.
* **Interface:** Modify the Gradio layout in `User_input.py` to adjust styling or add features.

---

## 🧪 Demo (Optional)

Include screenshots or GIFs of:

* Ingredient detection results on an image
* Retrieved Kannada recipe display
* Chatbot answering a cooking question in Kannada

---

## 🤝 Contributing

1. Fork this repository
2. Create your feature branch:

   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:

   ```bash
   git commit -m 'Add your feature'
   ```
4. Push to the branch:

   ```bash
   git push origin feature/your-feature
   ```
5. Open a Pull Request

---

## 🔒 Security Notes

* Never expose or commit API keys.
* Use `.gitignore` to exclude `.env` and private data.
* Rotate your keys regularly, especially if the project is public.

---

## 🙏 Acknowledgements

* Ultralytics YOLO — Object detection engine
* Gradio — Interactive app UI
* Google Gemini — Conversational AI
* ROBOFLOW — Open Kannada recipe datasets
