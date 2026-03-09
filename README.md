## Pneumonia Detection from Chest X‑ray (VGG19)

This repository contains a simple **Streamlit** web app for detecting pneumonia from chest X‑ray images using a **fine‑tuned VGG19** deep learning model.

- **GitHub repo**: [`working-with-big-files`](https://github.com/manishKrMahto/working-with-big-files.git)
- **Model file (VGG19 Fine‑Tuned model.h5)**: [`Google Drive link`](https://drive.google.com/file/d/1KGqQEnz2d4fDOdKxvRoUgJKlaTwCLLXM/view?usp=drivesdk)

The app loads the fine‑tuned VGG19 model and provides a browser UI where users can upload chest X‑ray images and receive a prediction (`NORMAL` or `PNEUMONIA`) with a confidence score.

---

### Project Structure

- `app.py` – Streamlit app that:
  - Loads the trained model from `VGG19 Fine-Tuned model.h5`
  - Accepts an uploaded chest X‑ray (`.jpg`, `.jpeg`, `.png`)
  - Preprocesses the image (resize to \(128 \times 128\), normalize, add batch dimension)
  - Runs inference and displays the predicted class and confidence
- `requirements.txt` – Python dependencies for running the app
- `VGG19 Fine-Tuned model.h5` – Fine‑tuned Keras model (download separately from Google Drive)

---

### Setup Instructions

#### 1. Clone the repository

```bash
git clone https://github.com/manishKrMahto/working-with-big-files.git
cd working-with-big-files
```

#### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS / Linux
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### Download and Place the Model File

1. Download the fine‑tuned VGG19 model from Google Drive:  
   [`VGG19 Fine-Tuned model.h5`](https://drive.google.com/file/d/1KGqQEnz2d4fDOdKxvRoUgJKlaTwCLLXM/view?usp=drivesdk)
2. Save the file into the **root of this repository** (same folder as `app.py`) with the **exact** filename:

```text
VGG19 Fine-Tuned model.h5
```

`app.py` expects this exact name and location when calling:

```python
model = load_model("VGG19 Fine-Tuned model.h5")
```

---

### Running the App

Once dependencies are installed and the model file is in place:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

---

### Usage

- Upload a chest X‑ray image in `.jpg`, `.jpeg`, or `.png` format.
- The app will:
  - Show the uploaded image
  - Run it through the VGG19 model
  - Display:
    - **Prediction**: `NORMAL` or `PNEUMONIA`
    - **Confidence score** in percentage

This project is meant as a demo for working with a relatively **large model file** (`.h5`) alongside a lightweight web interface.

---

### Requirements

All required Python packages are listed in `requirements.txt`. Key libraries:

- `streamlit` – for the web UI
- `tensorflow` / `keras` – to load and run the VGG19 model
- `Pillow` – for image handling
- `numpy` – for numerical processing
- `opencv-python` (`cv2`) – optional image utilities

