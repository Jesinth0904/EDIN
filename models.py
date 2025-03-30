import cv2
from deepface import DeepFace
import utils
import pygame
import os
import time

# Initialize pygame mixer for playing music
is_headless = "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ

try:
    if not is_headless:
        pygame.mixer.init()  # Initialize only if audio is available
    else:
        print("Running in a headless environment. Skipping pygame.mixer.init()")
except pygame.error as e:
    print(f"⚠️ Pygame mixer error: {e}")
    pygame.mixer.quit()
HAPPY_MUSIC_PATH = 'Our Cycle Bgm-Downringtone.com.mp3'  # Ensure this file exists in the project directory
music_played = False  # Track if music has been played


def play_music():
    global music_played
    if os.path.exists(HAPPY_MUSIC_PATH) and not music_played:
        pygame.mixer.music.load(HAPPY_MUSIC_PATH)
        pygame.mixer.music.play()
        music_played = True  # Mark that music has been played

def analyze_and_draw(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = utils.load_cascade()
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_region = frame[y:y+h, x:x+w]

        try:
            analysis = DeepFace.analyze(face_region, actions=['age', 'gender', 'emotion'], enforce_detection=False)
            analysis = analysis[0]
            
            age_text = f"Age: {analysis['age']}"
            gender_text = f"Gender: {analysis['dominant_gender']}"
            emotion_text = f"Emotion: {analysis['dominant_emotion']}"
            
            cv2.putText(frame, age_text, (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, gender_text, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, emotion_text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if analysis['dominant_emotion'] == 'happy':
                play_music()

        except Exception as e:
            print(f"Error analyzing face region: {e}")
    
    return frame

def stream_video(source):
    global music_played
    music_played = False  # Reset at the start of each video
    video_capture = cv2.VideoCapture(source)  
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  
        frame_with_analysis = analyze_and_draw(frame)
        cv2.imshow('Video Analysis', frame_with_analysis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
