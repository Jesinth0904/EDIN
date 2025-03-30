import cv2
from deepface import DeepFace
import utils
import pygame
import os
import time

# Initialize pygame mixer for playing music
pygame.mixer.init()
HAPPY_MUSIC_PATH = 'Our Cycle Bgm-Downringtone.com.mp3'  # Ensure this file exists in the project directory
music_start_time = None  # Track music play duration
music_played = False  # Track if music has already played

def play_music():
    global music_start_time, music_played
    if os.path.exists(HAPPY_MUSIC_PATH) and not music_played:
        if not pygame.mixer.music.get_busy():  # Only play if not already playing
            pygame.mixer.music.load(HAPPY_MUSIC_PATH)
            pygame.mixer.music.play()
            music_start_time = time.time()
            music_played = True  # Mark that music has played for this session

def stop_music():
    global music_start_time
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
        music_start_time = None

def analyze_and_draw(frame):
    global music_start_time, music_played
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = utils.load_cascade()
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    happy_detected = False

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
                happy_detected = True

        except Exception as e:
            print(f"Error analyzing face region: {e}")

    if happy_detected:
        if music_start_time is None:
            play_music()
        elif time.time() - music_start_time > 10:
            stop_music()
    else:
        stop_music()
    
    return frame

def stream_video(source):
    global music_played
    music_played = False  # Reset when a new video starts
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
    stop_music()  # Stop music when exiting
