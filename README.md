# AI_emotion_detection
This is a real-time facial emotion recognition system that detects and classifies human emotions through a webcam feed. The system uses a pre-trained deep learning model to identify seven distinct emotions and displays results with confidence scores, bounding boxes, and visual feedback.

Key Features:
Real-time detection - Processes webcam feed in real-time with minimal latency
7 emotion classes - Detects angry, disgust, fear, happy, neutral, sad, and surprise
Multiple face tracking - Can detect and analyze multiple faces simultaneously
Confidence visualization - Displays confidence bars and top emotion predictions
Ambiguous emotion handling - Shows both emotions when confidence scores are close (within 15% threshold)
Debug mode - Toggle detailed prediction probabilities for each face
Screenshot capture - Save annotated frames for later analysis

Computer Vision Pipeline:
Face Detection - Uses OpenCV's Haar Cascade classifier with optimized parameters for real-time performance
Face Preprocessing - Extracts face ROI, resizes to 48×48 pixels, converts to grayscale, and normalizes pixel values
Emotion Classification - TensorFlow/Keras deep learning model predicts emotion probabilities
Result Visualization - Draws colored bounding boxes, confidence bars, and text annotations

Model Architecture:
The system uses a convolutional neural network trained on facial expression datasets. The model outputs probability distribution across 7 emotion classes.
