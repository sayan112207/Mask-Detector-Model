import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the mask detection model
mask_model = load_model('mask_detection_model.h5')

# Define the classes that the model can detect
classes = ['with_mask', 'without_mask', 'partial_mask']

# Define the colors for the bounding boxes and labels
colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255)]

# Initialize the video stream
cap = cv2.VideoCapture(0)

# Loop over the frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    
    # If the frame was not read successfully, break the loop
    if not ret:
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use the face detection model to detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop over the detected faces and perform mask detection
    for (x, y, w, h) in faces:
        # Extract the face from the frame and resize it for mask detection
        face = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face, (224, 224))
        
        # Preprocess the face for the mask detection model
        normalized_face = resized_face / 255.0
        preprocessed_face = np.expand_dims(normalized_face, axis=0)
        
        # Make a prediction on the preprocessed face
        predictions = mask_model.predict(preprocessed_face)
        prediction = np.argmax(predictions[0])
        prediction_class = prediction
        prediction_confidence = predictions[0][prediction]
        
        # Add a bounding box and label to the frame
        color = colors[prediction_class]
        label = '{}: {:.2f}%'.format(classes[prediction_class], prediction_confidence*100)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Display the frame in a window
    cv2.imshow('Face and Mask Detector', frame)
    cv2.moveWindow('Face and Mask Detector', (1920 - 1350), (1080 - 1000))  # center the window
    
    # Wait for a key press and exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()