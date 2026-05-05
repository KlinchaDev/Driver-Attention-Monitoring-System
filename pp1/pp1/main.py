import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import face_recognition
import dlib
from scipy.spatial import distance
from pygame import mixer
import time
import os
import imutils
from imutils import face_utils
from ui2 import get_look_direction
class FaceEyeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face and Eye Detection")
        self.root.geometry("1000x600")  # Set a fixed window size

        self.image_path = ""
        self.video_path = ""
        self.cap = None
        self.frame_delay = 0

        # Load Haar cascades for face and eye detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Paths to known faces
        self.known_faces = {
            "Tatjana": r"C:\Users\klinc\Desktop\pp1\pp1\pp1\Projekt_z_sliki\slika_t.jpg",
            "Ana": r"C:\Users\klinc\Desktop\pp1\pp1\pp1\Projekt_z_sliki\slika_an.jpg",
            "Stefanija": r"C:\Users\klinc\Desktop\pp1\pp1\pp1\Projekt_z_sliki\slika_s.jpg"
        }

        # Load known face encodings
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

        # Initialize Dlib's face detector and create the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor_path = r"C:\Users\klinc\Desktop\pp1\pp1\pp1\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(self.shape_predictor_path):
            raise FileNotFoundError(f"The shape predictor file {self.shape_predictor_path} was not found. Please ensure the file is available.")
        self.predictor = dlib.shape_predictor(self.shape_predictor_path)

        # Initialize the mixer for playing alarm sound
        mixer.init()
        self.alarm_path = r"C:\Users\klinc\Desktop\pp1\pp1\pp1\alarm2.mp3"
        mixer.music.load(self.alarm_path)

        # Threshold values for EAR
        self.EYE_AR_THRESH = 0.23
        self.EYE_AR_CONSEC_FRAMES = 20
        self.EYE_CLOSE_DURATION = 3  # Duration in seconds for eyes closed to trigger alert

        # Initialize the frame counter and the total number of blinks
        self.counter = 0
        self.total_blinks = 0

        # Track the time when eyes were first detected closed
        self.close_start_time = None

        # Get the indexes of the facial landmarks for the left and right eye
        (self.left_start, self.left_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (self.right_start, self.right_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

        # Indices for nose tip and nose bridge
        self.nose_tip_idx = (30, 35)  # Points 30 to 35
        self.nose_bridge_idx = (27, 30)  # Points 27 to 30

        # Threshold for head orientation
        self.HEAD_ORIENTATION_THRESH = 15

        # Create and arrange widgets
        self.create_widgets()

    def create_widgets(self):
        # Create main frames
        left_frame = tk.Frame(self.root, width=200, padx=10, pady=10, bg="#f0f0f0")
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        right_frame = tk.Frame(self.root, padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Create and place buttons in the left frame
        self.upload_image_button = tk.Button(left_frame, text="Upload Image", command=self.upload_image, width=15, bg="#e7f3fe", fg="black")
        self.upload_image_button.pack(pady=10)

        self.upload_video_button = tk.Button(left_frame, text="Upload Video", command=self.upload_video, width=15, bg="#e7f3fe", fg="black")
        self.upload_video_button.pack(pady=10)

        self.detect_face_button = tk.Button(left_frame, text="Detect Face", command=self.detect_face, width=15, bg="#d1e7dd", fg="black")
        self.detect_face_button.pack(pady=10)

        self.detect_eyes_button = tk.Button(left_frame, text="Detect Eyes", command=self.detect_eyes, width=15, bg="#d1e7dd", fg="black")
        self.detect_eyes_button.pack(pady=10)

        self.detect_sunglasses_button = tk.Button(left_frame, text="Detect with Sunglasses", command=self.detect_with_sunglasses, width=15, bg="#fff3cd", fg="black")
        self.detect_sunglasses_button.pack(pady=10)

        self.canvas = tk.Canvas(right_frame, bg="white", relief=tk.SUNKEN, borderwidth=2)
        self.canvas.pack(expand=True, fill=tk.BOTH)

    def load_known_faces(self):
        for name, path in self.known_faces.items():
            image = face_recognition.load_image_file(path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
            else:
                messagebox.showerror("Error", f"Could not locate a face in the reference image for {name}.")

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            image = Image.open(self.image_path)
            image = image.resize((600, 600), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo
            messagebox.showinfo("Image Upload", "Image uploaded successfully!")

    def upload_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
               fps = 40  # Default to 30 if FPS cannot be determined
            self.frame_delay = int(1000 / (fps * 6.5)) #Speed increase by x6.5
            self.start_time = time.time()
            self.frame_count = 0
            self.start_blink_detection(video=True)

    def start_blink_detection(self, video=True):
        if not video:
            self.cap = cv2.VideoCapture(0)
            time.sleep(0.5)
        self.update_frame()

    def play_video(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = imutils.resize(frame, width=450)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray, 0)

                alarm_triggered = False

                for face in faces:
                    shape = self.predictor(gray, face)
                    shape = face_utils.shape_to_np(shape)
                    left_eye = shape[self.left_start:self.left_end]
                    right_eye = shape[self.right_start:self.right_end]
                    left_ear = self.eye_aspect_ratio(left_eye)
                    right_ear = self.eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0

                    if ear < self.EYE_AR_THRESH:
                        if self.close_start_time is None:
                            self.close_start_time = time.time()
                        eye_color = (0, 0, 255)  # Red for closed eyes
                        if time.time() - self.close_start_time >= self.EYE_CLOSE_DURATION:
                            cv2.putText(frame, "ALERT!!! Eyes closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 0, 255), 2)
                            alarm_triggered = True
                    else:
                        self.close_start_time = None
                        eye_color = (0, 255, 0)  # Green for open eyes

                    left_eye_hull = cv2.convexHull(left_eye)
                    right_eye_hull = cv2.convexHull(right_eye)
                    cv2.drawContours(frame, [left_eye_hull], -1, eye_color, 1)
                    cv2.drawContours(frame, [right_eye_hull], -1, eye_color, 1)

                    # Head Orientation Detection
                    landmarks = {
                        "left_eye": shape[self.left_start:self.left_end],
                        "right_eye": shape[self.right_start:self.right_end],
                        "nose_tip": shape[self.nose_tip_idx[0]:self.nose_tip_idx[1]],
                        "nose_bridge": shape[self.nose_bridge_idx[0]:self.nose_bridge_idx[1]]
                    }
                    look_direction = get_look_direction(landmarks)
                    print(f"Look Direction: {look_direction}")
                    cv2.putText(frame, f"Look Direction: {look_direction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)

                    if look_direction != "Looking Straight":
                        cv2.putText(frame, "ALERT!!! Not Looking Straight", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                        alarm_triggered = True

                if alarm_triggered:
                    if not mixer.music.get_busy():
                        mixer.music.play()
                else:
                    mixer.music.stop()

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (600, 600))
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.image = photo

                self.root.after(self.frame_delay, self.play_video)
            else:
                self.cap.release()

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture image")
                return

            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)

            alarm_triggered = False

            for face in faces:
                shape = self.predictor(gray, face)
                shape = face_utils.shape_to_np(shape)
                left_eye = shape[self.left_start:self.left_end]
                right_eye = shape[self.right_start:self.right_end]
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                if ear < self.EYE_AR_THRESH:
                    if self.close_start_time is None:
                        self.close_start_time = time.time()
                    eye_color = (0, 0, 255)  # Red for closed eyes
                    if time.time() - self.close_start_time >= self.EYE_CLOSE_DURATION:
                        cv2.putText(frame, "ALERT!!! Eyes closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                                    2)
                        alarm_triggered = True
                else:
                    self.close_start_time = None
                    eye_color = (0, 255, 0)  # Green for open eyes

                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, eye_color, 1)
                cv2.drawContours(frame, [right_eye_hull], -1, eye_color, 1)

                # Head Orientation Detection
                landmarks = {
                    "left_eye": shape[self.left_start:self.left_end],
                    "right_eye": shape[self.right_start:self.right_end],
                    "nose_tip": shape[self.nose_tip_idx[0]:self.nose_tip_idx[1]],
                    "nose_bridge": shape[self.nose_bridge_idx[0]:self.nose_bridge_idx[1]]
                }
                look_direction = get_look_direction(landmarks)
                print(f"Look Direction: {look_direction}")
                cv2.putText(frame, f"Look Direction: {look_direction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

                if look_direction != "Looking Straight":
                    cv2.putText(frame, "ALERT!!! Not Looking Straight", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)
                    alarm_triggered = True

            if alarm_triggered:
                if not mixer.music.get_busy():
                    mixer.music.play()
            else:
                mixer.music.stop()

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk

            self.root.after(10, self.update_frame)
        except Exception as e:
            print(f"Error: {e}")

    def detect_face(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        image = face_recognition.load_image_file(self.image_path)
        face_locations = face_recognition.face_locations(image, model="hog")
        face_encodings = face_recognition.face_encodings(image, face_locations)
        face_landmarks_list = face_recognition.face_landmarks(image)

        image = cv2.imread(self.image_path)
        unknown_person_detected = False
        for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown! Leave the Veichle!"
            color = (0, 0, 255)  # Red color for unknown faces

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                color = (0, 255, 0)  # Green color for known faces
            else:
                unknown_person_detected = True

            # Draw rectangle around face
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)

            # Calculate text size
            text_size, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
            text_w, text_h = text_size

            # Draw rectangle for text background
            cv2.rectangle(image, (left, top - text_h - 10), (left + text_w, top), color, -1)
            cv2.putText(image, name, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if unknown_person_detected:
            mixer.music.play()
            messagebox.showwarning("Warning", "Unknown person detected!")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((600, 600), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        messagebox.showinfo("Detection Complete", "Face detection and recognition complete.")

    def detect_eyes(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        eyes_detected = False
        eyes_closed = False
        for face in faces:
            shape = self.predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            left_eye = shape[self.left_start:self.left_end]
            right_eye = shape[self.right_start:self.right_end]
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < self.EYE_AR_THRESH:
                eye_color = (0, 0, 255)  # Red for closed eyes
                eyes_closed = True
            else:
                eyes_detected = True
                eye_color = (0, 255, 0)  # Green for open eyes

            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(image, [left_eye_hull], -1, eye_color, 1)
            cv2.drawContours(image, [right_eye_hull], -1, eye_color, 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((600, 600), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        if eyes_detected:
            messagebox.showinfo("Detection Complete", "Eye detection complete.")
        else:
            messagebox.showinfo("Detection Complete", "Eyes not found.")

        if eyes_closed:
            cv2.putText(image, "WARNING: EYES CLOSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            mixer.music.play()
            messagebox.showwarning("Warning", "Driver's eyes are closed!")

    def detect_with_sunglasses(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100))

        sunglasses_detected = False
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) == 0:
                sunglasses_detected = True
                cv2.putText(image, "WARNING: SUNGLASSES DETECTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
                mixer.music.play()

            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if not sunglasses_detected:
            cv2.putText(image, "Sunglasses not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((600, 600), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        if sunglasses_detected:
            messagebox.showwarning("Warning", "Sunglasses detected!")
        else:
            messagebox.showinfo("Detection Complete",
                                "Face detection with sunglasses complete. Sunglasses not detected.")

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceEyeDetectionApp(root)
    root.mainloop()
