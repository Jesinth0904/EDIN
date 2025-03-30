import os
import cv2 

def save_uploaded_video(uploaded_file):
    video_path = os.path.join('uploads', 'input.mp4')

    with open(video_path, 'wb') as wf:
        wf.write(uploaded_file.getbuffer())

    return video_path


def load_cascade():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade