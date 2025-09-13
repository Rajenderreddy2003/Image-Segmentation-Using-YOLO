````markdown
# Image Segmentation App (YOLOv11x-seg)

This is an interactive Object Segmentation Web App built with Streamlit and powered by YOLOv11x-seg (Ultralytics).  
It allows you to perform segmentation on any uploaded image or directly from your camera.  

---

## Features
- Upload any image (`jpg`, `jpeg`, `png`, `webp`)  
- Use your camera to capture and segment objects instantly  
- Visualize segmented objects with colored masks  
- Powered by YOLOv11x-seg, a state-of-the-art instance segmentation model  
- Deployed on Hugging Face Spaces using Streamlit  

---

## How It Works
1. Load the YOLOv11x-seg model pipeline (stored as a `yolo_pipeline.pkl` file).  
2. Accept input from the user:
   - Upload image  
   - Or capture from the camera  
3. Run YOLOv11x segmentation to detect and segment objects.  
4. Overlay random colors on each detected object.  
5. Display the segmented output image.  

---

## Installation & Running Locally

### 1. Clone this repository
```bash
git clone https://huggingface.co/spaces/YourUsername/Object-Segmentation-App
cd Object-Segmentation-App
````

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

### 5. Open in browser

Go to [http://localhost:8501](http://localhost:8501)

---

## Project Structure

```
Object-Segmentation-App/
│── pipeline.py          # YOLO pipeline class
│── save_pickle.py       # Script to generate yolo_pipeline.pkl
│── yolo_pipeline.pkl    # Saved pipeline with YOLOv11x-seg model
│── streamlit_app.py     # Streamlit web app
│── requirements.txt     # Dependencies
│── README.md            # Project documentation
```

---

## Deployment on Hugging Face

1. Create a new Hugging Face Space (select Streamlit template).
2. Upload all project files (`pipeline.py`, `streamlit_app.py`, `requirements.txt`, `yolo_pipeline.pkl`).
3. Push to Hugging Face repo:

   ```bash
   git add .
   git commit -m "Initial commit"
   git push
   ```
4. Hugging Face will auto-build and deploy your app.

---

## Tech Stack

* [YOLOv11x-seg (Ultralytics)](https://docs.ultralytics.com)
* [Streamlit](https://streamlit.io/)
* [OpenCV](https://opencv.org/)
* [NumPy](https://numpy.org/)
* [Pillow](https://python-pillow.org/)

```
