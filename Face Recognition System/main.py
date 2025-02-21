import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from deepface import DeepFace

KNOWN_FACES_DIR = "known_faces"
CAPTURED_IMAGE = "captured.jpg"

# Dictionary to store face encodings
known_faces = {}

# Load all known faces
for person_name in os.listdir(KNOWN_FACES_DIR):
    person_path = os.path.join(KNOWN_FACES_DIR, person_name)

    if os.path.isdir(person_path):
        encodings = []

        for image_name in os.listdir(person_path):
            image_path = os.path.abspath(os.path.join(person_path, image_name))

            try:
                embedding = DeepFace.represent(image_path, model_name="ArcFace")[0]['embedding']
                encodings.append(np.array(embedding))
                print(f"✅ Loaded: {image_path}")
            except Exception as e:
                print(f"❌ Error: {e}")

        if encodings:
            known_faces[person_name] = encodings

print("✅ All known faces loaded!")


# Function to capture a photo
def capture_photo():
    cap = cv2.VideoCapture(1)
    cv2.waitKey(1000)
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(CAPTURED_IMAGE, frame)
        display_image(CAPTURED_IMAGE)
        result_label.config(text="📸 Photo Captured!")


# Function to compare with known faces
def recognize_face():
    try:
        embedding = DeepFace.represent(CAPTURED_IMAGE, model_name="ArcFace")[0]['embedding']
    except:
        result_label.config(text="❌ No Face Detected!")
        return

    best_match = None
    best_score = 0.5  # Minimum threshold

    for name, encodings in known_faces.items():
        for known_embedding in encodings:
            score = np.dot(embedding, known_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(known_embedding))
            if score > best_score:
                best_match = name
                best_score = score

    if best_match:
        result_label.config(text=f"✅ Access Granted: {best_match}", fg="green")
    else:
        result_label.config(text="❌ Access Denied", fg="red")


# Function to display an image in Tkinter
def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((200, 200))  # Resize for UI
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img


# ---------------------- Tkinter UI ---------------------- #
root = tk.Tk()
root.title("Face Recognition System")
root.geometry("400x500")

# Title
title_label = Label(root, text="🔒 Face Recognition", font=("Arial", 16))
title_label.pack(pady=10)

# Image display
image_label = Label(root)
image_label.pack()

# Capture button
capture_button = Button(root, text="📸 Capture Photo", command=capture_photo)
capture_button.pack(pady=10)

# Recognize button
recognize_button = Button(root, text="🔍 Recognize Face", command=recognize_face)
recognize_button.pack(pady=10)

# Result label
result_label = Label(root, text="Status: Waiting...", font=("Arial", 14))
result_label.pack(pady=10)

# Run the Tkinter loop
root.mainloop()
