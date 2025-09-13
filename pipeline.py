# pipeline.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pickle


class YoloSegmentationPipeline:
    def __init__(self, model_path="yolo11x-seg.pt"):
        # Load YOLO segmentation model
        self.model = YOLO(model_path)

    def load_and_show_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def predict(self, image):
        # Run YOLO segmentation
        results = self.model.predict(source=image, save=False, save_txt=False)
        return results

    def visualize_segmentation(self, results):
        """Overlay segmentation masks on the original image."""
        result = results[0]
        masks = result.masks.data  # Segmentation masks
        image = result.orig_img.copy()

        # Create overlay
        mask_overlay = np.zeros_like(image, dtype=np.uint8)
        for mask in masks:
            mask_binary = mask.cpu().numpy().astype(np.uint8)
            mask_binary = cv2.resize(mask_binary, (mask_overlay.shape[1], mask_overlay.shape[0]))

            color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
            mask_overlay[mask_binary > 0] = color

        combined_image = cv2.addWeighted(image, 0.5, mask_overlay, 0.5, 0)
        return cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
    

# Initialize pipeline
pipeline = YoloSegmentationPipeline("yolo11x-seg.pt")

# Save pipeline to pickle
with open("yolo_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Pipeline saved as yolo_pipeline.pkl")
