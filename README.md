# ♻️ AI Waste Classification System (Zero-Shot Learning)

An AI-powered web application that helps users identify how to dispose of waste items correctly.  
Users upload an image of waste, and the system analyzes it using a **pre-trained OpenAI CLIP model** to suggest the appropriate disposal category such as **Plastic, Paper, Glass, Hazardous, E-Waste, Food Waste, or Landfill**.

This project uses **zero-shot image classification**, meaning **no dataset training is required**.

---

## 🚀 Project Features

- 📸 Upload an image of a waste item
- 🤖 AI-based classification using a pre-trained model
- 🗑️ Suggests the correct disposal bin
- 📊 Shows confidence score and top-3 predictions
- ⚠️ Handles uncertain or mixed waste cases
- 🌍 Designed with real-world recycling awareness in mind

---

## 🧠 Technology Stack

### Artificial Intelligence / Machine Learning
- OpenAI **CLIP** model (via Hugging Face)
- Zero-shot image classification
- PyTorch (model inference)

### Backend
- Python
- Flask (web framework)

### Computer Vision
- Pillow (PIL) for image processing

### Frontend
- HTML5
- CSS3 (basic styling)

### Tools & Platform
- Git & GitHub
- Python Virtual Environment
- Works on Windows / Linux / macOS
- Deployable on Raspberry Pi (CPU-based)

---

## 📂 Project Structure

```text
AI-Waste-Classification-System/
│
├── app.py
├── requirements.txt
├── README.md
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── style.css
│   └── uploads/
│
└── .venv/   (ignored in Git)

Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/Rajanelluri/AI-Waste-Classification-System.git
cd AI-Waste-Classification-System

2️⃣ Create and activate virtual environment (Python 3.11 recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the application
python app.py

5️⃣ Open in browser
http://127.0.0.1:5000
