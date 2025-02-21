# Face Recognition System

## 📌 Description
This is a simple Face Recognition System using OpenCV, DeepFace, and Tkinter. The application captures a photo using a webcam, compares the face with known faces stored in a directory, and grants or denies access based on recognition results.

## 🚀 Features
- Load and store known face encodings.
- Capture a new photo using the webcam.
- Recognize the captured face by comparing it with stored encodings.
- Display results in a user-friendly Tkinter interface.

## 🛠️ Requirements
Ensure you have the following dependencies installed before running the application:

```bash
pip install opencv-python numpy pillow deepface tkinter
```

## 📂 Project Structure
```
.
├── known_faces/         # Directory containing folders of known individuals
│   ├── Person1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── Person2/
│   │   ├── image1.jpg
├── captured.jpg         # Last captured photo (temporary)
├── face_recognition.py  # Main application script
└── README.md            # Project documentation
```

## 🔧 How to Use
1. Place images of known individuals in `known_faces/{person_name}/`.
2. Run the script:
   ```bash
   python face_recognition.py
   ```
3. Click `📸 Capture Photo` to take a picture using your webcam.
4. Click `🔍 Recognize Face` to compare the captured photo with stored faces.
5. If a match is found, access is granted; otherwise, it is denied.

## 🛑 Troubleshooting
- If no face is detected, ensure your webcam is working and the image has proper lighting.
- If recognition fails, make sure known faces are clear and have good resolution.

## 🏆 Credits
- OpenCV for image processing
- DeepFace for face embedding and recognition
- Tkinter for GUI

## 📜 License
This project is open-source and free to use under the MIT License.

---
🎯 **Happy Coding!** 🚀

