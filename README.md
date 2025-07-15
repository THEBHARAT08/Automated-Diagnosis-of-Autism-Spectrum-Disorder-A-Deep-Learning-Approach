
# 🧠 Automated Diagnosis of Autism Spectrum Disorder: A Deep Learning Approach

This project automates the detection of Autism Spectrum Disorder (ASD) from facial images using a deep learning model (InceptionV3). A web interface is provided using Flask to make the model accessible for inference.

---

## 📁 Project Structure

```
├── app.py                          # Flask web app to handle UI and prediction
├── Autism_original.ipynb          # Jupyter Notebook used for model development
├── requirements.txt               # Required Python libraries
├── InceptionV3_model.h5           # Pre-trained InceptionV3 model
├── haarcascade_frontalface_*.xml  # OpenCV face detection file
├── templates/                     # HTML templates (index.html, result.html)
├── uploads/                       # Folder for uploaded images
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/autism-diagnosis.git
cd autism-diagnosis
```

### 2. Create and Activate Virtual Environment

#### On macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Application

```bash
python app.py
```

Then, open your browser and go to:  
👉 `http://127.0.0.1:5000/`

---

## 🔍 How It Works

1. Upload a facial image on the web interface.
2. The system detects the face using Haarcascade.
3. The image is processed and passed to the InceptionV3 model.
4. The model predicts whether the input image indicates signs of ASD.

---

## 🧠 Model Details

- **Model:** InceptionV3
- **Framework:** TensorFlow/Keras
- **File:** `InceptionV3_model.h5`
- **Face Detection:** Haarcascade (`.xml`)

---

## 📌 Notes

- Ensure `InceptionV3_model.h5` and `haarcascade_frontalface_default.xml` are in the root directory.
- Place your HTML UI files inside the `templates/` folder.
- Uploaded images are stored temporarily in `uploads/`.

---

## 📊 Data Source

The dataset used for training and evaluation is available on Kaggle:  
🔗 [Autism Image Data - Kaggle](https://www.kaggle.com/datasets/cihan063/autism-image-data)

---

## 📃 License

This project is intended for academic and educational purposes only.
