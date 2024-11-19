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

"""--------------------------------------------------------------------------------------------------------------
# PART 0
-----------------------------------------------------------------------------------------------------------------
Sets initial parameters.

"""
# temp admin area
textExample = [(50,50),datetime.now(),"3 x pers with AK47 showing hostile intent, mounted in white rangerover","en",None]
newTxObjects = [textExample]


# Load YOLOv8 model
model = YOLO('./yolov8models/yolov8s-visdrone.pt')  # Use a lightweight model for real-time performance

# resolution calibration variables
confidenceThreshold = 0.5 #varies between 0 and 1
spatialThreshold = 250 # in transformed coordinates
ageOff = 5 # in seconds


# plot variables
iconSize = 0.05
groundX = 500 # width in cm of ground square
groundY = 500 # heigh in cm of ground square
visibleCorners = [[0,0],[groundX,0],[0,groundY],[groundX,groundY]] #bl, br, tl, tr

# Initialize webcam
cap = cv2.VideoCapture(1)  # 0 is the default webcam

# Initialize tracked objects and Matplotlib data
trackedObjects = {} # [id]:[(x,y),TOI, description,identification(friend, enemy, unk), [file_loc]]
agedTrackedObjects = {}
object_count = 0  # Unique ID for each detected object
objTOI = datetime.now()
x_data, y_data = [], []


# Set up Matplotlib for live plotting
# NEED TO FIX THIS AND ENSURE IT LINES UP WITH F1

fig, ax = plt.subplots()
sc = ax.scatter([], [], color='red')
ax.set_xlim(0, groundX)  # Set limits based on frame size
ax.set_ylim(0, groundY)
ax.invert_yaxis()  # Invert y-axis to match OpenCV's coordinate system

#----------------------------------------------------------------------------------------------------------------

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
    imageObjects = []

    for result in results:
        for box in result.boxes:
            
            # GET NEW OBJECT
            centreX, centreY, label, confidence, x1, y1, x2, y2 = F1(box, model, frame, visibleCorners)
            imObjdId = object_count
            imageObjects.append((imObjdId, (centreX, centreY), label, confidence, (x1, y1, x2, y2)))
            object_count += 1

    # RESOLVE NEW OBJECTS WITH KNOWN OBJECTS
    newImObjects = imageObjects.copy()
    for trackedObjId, imEntity in trackedObjects.items():
        prevX, prevY = imEntity[0]  # position of current tracked entity 

        # IMAGERY OBJECT RESOLUTION
        for obj_id, center, label, confidence, bbox in newImObjects: # iterate over known list of entities

            imDistance = ((centreX - prevX) ** 2 + (centreY - prevY) ** 2) ** 0.5
            if imDistance < spatialThreshold and confidence > confidenceThreshold:  
                trackedObjects[trackedObjId][0] = (centreX, centreY) #(x,y)
                trackedObjects[trackedObjId][1] = objTOI #TOI
                trackedObjects[trackedObjId][2] += " " + label + " detected by camera 1 @ " + str(objTOI) #description,identification(friend, enemy, unk), [file_loc]]
                newImObjects.remove((obj_id, center, label, confidence, bbox)) # remove item from list if resolved
        
        # TEXT OBJECT RESOLUTION
        #[(x,y),TOI, description,identification(friend, enemy, unk), [file_loc]]
        for textEnt in newTxObjects:
            centreX, centreY = textEnt[0]
            txDistance = ((centreX - prevX) ** 2 + (centreY - prevY) ** 2) ** 0.5
            if txDistance < spatialThreshold:
                # only update TOI, description and enemy. Assume that spatial info is more accurate from camera.
                trackedObjects[trackedObjId][1] = textEnt[1] #TOI
                trackedObjects[trackedObjId][2] += f" @{textEnt[1]}: {textEnt[2]}"
                newTxObjects.remove(textEnt)

    # ADD NEW IMAGE OBJECTS 
    for obj_id, center, label, confidence, bbox in newTxObjects:
        if confidence > confidenceThreshold:
            trackedObjects[obj_id] = [(centreX, centreY),objTOI,label,None,[None]]
                            
    # ADD NEW TEXT OBJECTS
    for obj_id, center, label, confidence, bbox in newImObjects:
        if confidence > confidenceThreshold:
            trackedObjects[obj_id] = [(centreX, centreY),objTOI,label,None,[None]]

    #-------------------------------------------------------------------------------------------------------
    # AGE OFF TRACKED OBJECTS
    #-------------------------------------------------------------------------------------------------------
    current_time = datetime.now()
    agedTrackedObjects = {key: ent for key, ent in trackedObjects.items() if (current_time-ent[1]).total_seconds() <= ageOff}

    #-------------------------------------------------------------------------------------------------------
    # VIDEO PLOT
    #-------------------------------------------------------------------------------------------------------
    # Draw bounding boxes, labels, and IDs
    for obj_id, center, label, confidence, bbox in imageObjects:
        with open("debug1.txt","a") as f:
            f.write(f"{object_count} {len(imageObjects)}\n")

        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
        cv2.putText(frame, f'{label} {confidence:.2f} ID: {obj_id} Len: {len(trackedObjects)}', (x1, y1 - 10),
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
    for ent in agedTrackedObjects.values():

        # apply correct mil symbol (only enemy and unknown available)
        if ent[3] == "en":
            iconImage = OffsetImage(enIcon, zoom=iconSize)
        else:
            iconImage = OffsetImage(unkIcon, zoom=iconSize)

        # create mil symbol
        annotationText = f"{ent[0]}"
        ab = AnnotationBbox(iconImage, ent[0], frameon=False)
        ax.add_artist(ab)
             

        # clear old text
        try:
            plotText.remove()
            scatterText.remove()
        except:
            pass

        # add text to entities
        text_offset_x = 0.5  # Adjust the offset as needed
        text_offset_y = 0.5  # Adjust the offset as needed
         
        scatterText = ax.text(ent[0][0] + text_offset_x, ent[0][1] + text_offset_y, annotationText, fontsize=12, color='black')

        # add text to plot
        tracked_object_count = len(agedTrackedObjects)
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
