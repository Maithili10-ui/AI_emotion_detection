import cv2
import numpy as np
from tensorflow import keras
import os

class EmotionDetector:
    def __init__(self, model_path, emotions):
        self.model = keras.models.load_model(model_path)
        self.emotions = emotions
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def preprocess_face(self, face_roi):
        """Preprocess face for emotion prediction"""
        # Resize to 48x48 and convert to grayscale
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = face_roi.astype('float32') / 255.0
        face_roi = np.expand_dims(face_roi, axis=-1)  # Add channel dimension
        face_roi = np.expand_dims(face_roi, axis=0)   # Add batch dimension
        return face_roi
    
    def detect_emotion(self, frame):
        """Detect faces and predict emotions in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with more sensitive parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # More sensitive
            minNeighbors=3,    # More sensitive
            minSize=(50, 50)   # Larger minimum size for better quality
        )
        
        results = []
        
        for (x, y, w, h) in faces:
            # Extract face ROI with padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            face_roi = frame[y1:y2, x1:x2]
            
            # Preprocess for emotion detection
            processed_face = self.preprocess_face(face_roi)
            
            # Predict emotion
            predictions = self.model.predict(processed_face, verbose=0)
            
            # Get top 2 emotions and their confidences
            top2_indices = np.argsort(predictions[0])[-2:][::-1]
            emotion1 = self.emotions[top2_indices[0]]
            confidence1 = predictions[0][top2_indices[0]]
            emotion2 = self.emotions[top2_indices[1]]
            confidence2 = predictions[0][top2_indices[1]]
            
            # If top 2 emotions are close in confidence, show both
            if abs(confidence1 - confidence2) < 0.15:  # 15% threshold
                emotion = f"{emotion1}/{emotion2}"
                confidence = max(confidence1, confidence2)
            else:
                emotion = emotion1
                confidence = confidence1
            
            results.append({
                'bbox': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence,
                'all_predictions': predictions[0]  # Store all predictions for debugging
            })
            
            # Draw bounding box and emotion text
            color = self.get_emotion_color(emotion1)  # Use primary emotion for color
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Emotion text with confidence - show both if close
            if "/" in emotion:
                text = f"{emotion}: {confidence:.2f}"
            else:
                text = f"{emotion}: {confidence:.2f}"
                
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Confidence bar
            bar_width = w
            bar_height = 8
            confidence_width = int(bar_width * confidence)
            cv2.rectangle(frame, (x, y+h+5), (x+confidence_width, y+h+5+bar_height), color, -1)
            cv2.rectangle(frame, (x, y+h+5), (x+bar_width, y+h+5+bar_height), color, 1)
            
            # Show secondary emotion if confidence is close
            if abs(confidence1 - confidence2) < 0.15:
                secondary_text = f"or {emotion2}: {confidence2:.2f}"
                cv2.putText(frame, secondary_text, (x, y+h+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame, results
    
    def get_emotion_color(self, emotion):
        """Get color based on emotion"""
        colors = {
            'happy': (0, 255, 0),      # Green
            'neutral': (255, 255, 0),  # Yellow
            'surprise': (255, 0, 255), # Magenta
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'fear': (0, 165, 255),     # Orange
            'disgust': (0, 255, 255)   # Cyan
        }
        return colors.get(emotion, (255, 255, 255))  # Default white

def main():
    # Emotion labels (must match training)
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Initialize detector
    detector = EmotionDetector('models/emotion_model.h5', emotions)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("🎭 Enhanced Real-time Emotion Detection Started!")
    print("📝 Instructions:")
    print("   - Make exaggerated expressions for better detection")
    print("   - 😢 Sad: Downturned mouth, furrowed brows")
    print("   - 😠 Angry: Tight lips, narrowed eyes") 
    print("   - 😲 Surprised: Wide eyes, raised brows")
    print("   - Press 'q' to quit, 's' to save screenshot")
    print("   - Press 'd' to show debug info")
    
    show_debug = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect emotions
        processed_frame, results = detector.detect_emotion(frame)
        
        # Show debug information if enabled
        if show_debug and results:
            for i, result in enumerate(results):
                debug_y = 30 + (i * 120)
                cv2.putText(processed_frame, f"Face {i+1} Predictions:", (10, debug_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                for j, (emotion, confidence) in enumerate(zip(emotions, result['all_predictions'])):
                    color = (0, 255, 0) if confidence == max(result['all_predictions']) else (255, 255, 255)
                    text = f"  {emotion}: {confidence:.3f}"
                    cv2.putText(processed_frame, text, (10, debug_y + 15 + (j * 15)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Display frame count and results summary
        cv2.putText(processed_frame, f"Faces detected: {len(results)}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if show_debug:
            cv2.putText(processed_frame, "DEBUG MODE - Press 'd' to hide", (10, processed_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Enhanced Emotion Detection - Press Q to quit', processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            cv2.imwrite('emotion_screenshot.png', processed_frame)
            print("💾 Screenshot saved as 'emotion_screenshot.png'")
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"🔧 Debug mode: {'ON' if show_debug else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Emotion detection ended!")

if __name__ == "__main__":
    main()