import logging
from toyMaven.spatialFunctions import F1
from ultralytics import YOLO
import cv2

def cameraObjects(queue, visibleCorners, cameraChoice = 1):

    # Suppress logging for YOLOv8
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    model = YOLO('./yolov8models/yolov8s-visdrone.pt')  # Use a lightweight model for real-time performance

    # Initialize webcam
    cap = cv2.VideoCapture(cameraChoice)  # 0 is the default webcam

    # Initialize tracked objects and Matplotlib data
    object_count = 0  # Unique ID for each detected object

    #----------------------------------------------------------------------------------------------------------------
    # Main loop for object detection and tracking
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model.predict(frame)

        # Clear previous center points for new frame
        imageObjects = []
        
        for result in results:
            for box in result.boxes:
                
                # GET NEW OBJECT
                centreX, centreY, label, confidence, x1, y1, x2, y2 = F1(box, model, frame, visibleCorners)
                imObjdId = object_count
                imageObject = (imObjdId, (centreX, centreY), label, confidence, (x1, y1, x2, y2))
                imageObjects.append(imageObject)
                object_count += 1

        # PASS NEW OBJECTS TO QUEUE
        queue.put(("image", imageObjects, frame))
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
  