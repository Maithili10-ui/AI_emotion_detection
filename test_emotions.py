# test_emotions.py
import cv2
import numpy as np
from tensorflow import keras

def test_specific_emotions():
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    model = keras.models.load_model('models/emotion_model.h5')
    
    cap = cv2.VideoCapture(0)
    
    print("🎭 Emotion Detection Test")
    print("Make these expressions and see the confidence scores:")
    print("😊 Happy - 😢 Sad - 😠 Angry - 😲 Surprised")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            # Draw face box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Process face for emotion detection
            face_roi = frame[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = np.expand_dims(face_roi, axis=-1)
            face_roi = np.expand_dims(face_roi, axis=0)
            
            # Get all emotion predictions
            predictions = model.predict(face_roi, verbose=0)[0]
            
            # Display all emotion confidences
            y_offset = y - 10
            for i, (emotion, confidence) in enumerate(zip(emotions, predictions)):
                color = (0, 255, 0) if confidence == max(predictions) else (255, 255, 255)
                text = f"{emotion}: {confidence:.3f}"
                cv2.putText(frame, text, (x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset -= 15
        
        cv2.imshow('Emotion Confidence Test - Press Q to quit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_specific_emotions()