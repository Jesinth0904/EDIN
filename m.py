import cv2
from deepface import DeepFace
import utils


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
        except Exception as e:
            print(f"Error analyzing face region: {e}")

    return frame


def stream_video(source):

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
