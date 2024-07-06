import cv2
import os
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Path to images
cat_images_path = 'drive/MyDrive/Cat'

# Haar cascade for cat face detection
cat_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalcatface.xml'
if not os.path.exists(cat_cascade_path):
    raise FileNotFoundError("Cat face Haar cascade file not found.")
cat_cascade = cv2.CascadeClassifier(cat_cascade_path)

# Haar cascade for human face detection
human_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(human_cascade_path):
    raise FileNotFoundError("Human face Haar cascade file not found.")
human_cascade = cv2.CascadeClassifier(human_cascade_path)

# Load pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.COCO_V1")
model.eval()

# Transform for object detection
transform = T.Compose([
    T.ToTensor()
])

for image_name in os.listdir(cat_images_path):
    image_path = os.path.join(cat_images_path, image_name)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Skipping {image_name}: unable to load image.")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect cat faces
    cat_faces = cat_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2)

    # Detect human faces
    #human_faces = human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4), {len(human_faces)} human faces"

    print(f"Processing {image_name}: {len(cat_faces)} cat faces")

    # Check if exactly one cat face is detected
    if len(cat_faces) == 1:
        # Convert image to PIL format for object detection
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_tensor = transform(pil_image).unsqueeze(0)

        # Perform object detection
        with torch.no_grad():
            detections = model(image_tensor)[0]

        # Check if there are any objects other than cats with more than 0.5 confidence
        non_cat_objects_detected = False
        for label, score in zip(detections['labels'], detections['scores']):
            if score > 0.5 and label != 17:  # 17 is the COCO label for cat
                non_cat_objects_detected = True
                break

        if non_cat_objects_detected:
            print(f"Discarding {image_name} because non-cat objects were detected")
            os.remove(image_path)
        else:
            print(f"Keeping {image_name}")
    else:
        print(f"Discarding {image_name} due to incorrect face count")
        os.remove(image_path)
