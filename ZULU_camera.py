# NOV 24 - INT CORPS AI BRIEF MAVEN EXAMPLE
'''
This file orchestrates the detection of a remote control car and it's transformation into a target. 
It's purpose is to demonstrate how the generation of a red picture (and intelligence corps sport) need:
- a good understanding of the problem you're trying to solve.
- constant collaboration with people from different capbadges/militaries.
- a working knowledge of the systems you need to use, and what processes you need to go through to connect to / access them.
- what the technology does and doesn't do
'''


import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import deque
from toyMaven.spatialFunctions import F1
from datetime import datetime, timedelta
from matplotlib.offsetbox import OffsetImage, AnnotationBbox



# Load YOLOv8 model
model = YOLO('./yolov8models/yolov8s-visdrone.pt')  # Use a lightweight model for real-time performance

# resolution calibration variables
confidenceThreshold = 0.5 #varies between 0 and 1
spatialThreshold = 150 # in transformed coordinates
ageOff = 5 # in seconds

# plot variables
iconSize = 0.05

# Initialize webcam
cap = cv2.VideoCapture(1)  # 0 is the default webcam

# Initialize tracked objects and Matplotlib data
tracked_objects = {} # [id]:[(x,y),TOI, description,identification(friend, enemy, unk), [file_loc]]
aged_tracked_objects = {}
object_count = 0  # Unique ID for each detected object
objTOI = datetime.now()
x_data, y_data = [], []


# Set up Matplotlib for live plotting
fig, ax = plt.subplots()
sc = ax.scatter([], [], color='red')
ax.set_xlim(0, 640)  # Set limits based on frame size
ax.set_ylim(0, 480)
ax.invert_yaxis()  # Invert y-axis to match OpenCV's coordinate system

# Main loop for object detection and tracking
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model.predict(frame)
    objTOI = datetime.now()

    # Clear previous center points for new frame
    x_data.clear()
    y_data.clear()
    current_objects = []

    for result in results:
        for box in result.boxes:
            
            # Get data from model results
            center_x, center_y, label, confidence, x1, y1, x2, y2 = F1(box, model, frame)
            #-------------------------------------------------------------------------------------------------------

            #-------------------------------------------------------------------------------------------------------
            'F2: TRACK / RESOLVE OBJECTS'
            ' a. track objects'
            ' b. store objects and recall'
            ' if we are storing objects and recalling them for use here, then we can run a seperate process to enrich those objects with other data'

            # Track objects based on proximity to previous frames and threshold
            
            # CONFIDENCE
            if  confidence > confidenceThreshold:
                object_found = False
                for obj_id, prev_center in tracked_objects.items():
                    prev_x, prev_y = prev_center[0]  # Last known position
                    distance = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
                    
                    # DISTANCE 
                    if distance < spatialThreshold:  # Adjust threshold as needed
                        tracked_objects[obj_id][0] = (center_x, center_y) #(x,y)
                        tracked_objects[obj_id][1] = objTOI #TOI
                        #tracked_objects[obj_id][2] = label #description,identification(friend, enemy, unk), [file_loc]]
                        object_found = True
                        current_objects.append((obj_id, (center_x, center_y), label, confidence, (x1, y1, x2, y2)))
                        break

                # NEW OBJECT
                if not object_found:  # New object
                    tracked_objects[object_count] = [(center_x, center_y),objTOI,label,None,[None]]
                    current_objects.append((object_count, (center_x, center_y), label, confidence, (x1, y1, x2, y2)))
                    object_count += 1
                

            #-------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------
    # AGE OFF
    #-------------------------------------------------------------------------------------------------------
    current_time = datetime.now()
    if len(tracked_objects) > 0:
        [print(ii[1]) for key, ii in tracked_objects.items()]
        aged_tracked_objects = {key: ent for key, ent in tracked_objects.items() if (current_time-ent[1]).total_seconds() <= ageOff}

    #-------------------------------------------------------------------------------------------------------
    # VIDEO PLOT
    #-------------------------------------------------------------------------------------------------------
    # Draw bounding boxes, labels, and IDs
    for obj_id, center, label, confidence, bbox in current_objects:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
        cv2.putText(frame, f'{label} {confidence:.2f} ID: {obj_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    #-------------------------------------------------------------------------------------------------------
    # ACTUAL PLOT
    #-------------------------------------------------------------------------------------------------------
    # Update scatter plot with new points. Note it will error out if there are no objects detected without if statement
    import matplotlib.image as mpimg
    enIcon = mpimg.imread("./icons/en.png")
    unkIcon = mpimg.imread("./icons/unk.png")

    # clear scatter plot
    for artist in ax.artists:  # ax.artists contains all added AnnotationBbox objects
        artist.remove()
        
    # go through each object and plot
    for ent in aged_tracked_objects.values():
        if ent[3] == "en":
            iconImage = OffsetImage(enIcon, zoom=iconSize)
        else:
            iconImage = OffsetImage(unkIcon, zoom=iconSize)

        ab = AnnotationBbox(iconImage, ent[0], frameon=False)
        ax.add_artist(ab)
             
        
        # clear old text
        try:
            plotText.remove()
        except:
            pass

        # add new text
        tracked_object_count = len(aged_tracked_objects)
        plotText = ax.text(0.5, 0.9, f'No. of Tracked Objects: {tracked_object_count}', transform=ax.transAxes,
        fontsize=12, verticalalignment='top', horizontalalignment='center')
    plt.pause(0.01)  # Pause to update the plot
    

    # Show webcam feed
    cv2.imshow("Webcam Object Tracking", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
plt.close(fig)
