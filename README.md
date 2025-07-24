# 🥔 Potato Disease Classification

This project is a machine learning-based system that classifies potato leaf diseases into one of the following categories:

- ✅ Healthy
- ✅ Early Blight
- ✅ Late Blight

The system uses a **Convolutional Neural Network (CNN)** trained on potato leaf images. It includes a **React frontend** for image upload and a **Python backend API** that returns the predicted class along with the confidence level.

---

## 📸 Example Output

## 🧠 Model Overview

- Built using **TensorFlow/Keras**
- Architecture: Simple CNN with Conv2D, MaxPooling, Dropout, and Dense layers
- Output: Softmax probabilities over 3 classes
- Trained on labeled images of potato leaves

## 🗂 Project Structure

potato-disease-classification/
├── api/                   # Python backend (Flask or FastAPI)
│   └── main.py            # Handles image processing & prediction
│
├── frontend/              # React frontend
│   └── (components and pages)
│
├── training/              # Model training notebook and dataset
│   ├── train.ipynb
│   └── PlantVillage/      # Dataset folder
│
├── models/                # Saved model weights
├── models.config          # Model config (optional)
├── package.json           # Frontend dependencies
├── README.md              # Project documentation
└── .gitignore             # Files to ignore in Git


## 🚀 How to Run

### 1️⃣ Backend Setup (API)

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8001

### 2 frontend Setup (API)
cd frontend
npm install
npm start
