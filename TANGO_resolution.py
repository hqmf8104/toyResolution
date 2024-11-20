
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import deque
from toyMaven.spatialFunctions import F1
from datetime import datetime, timedelta
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import logging

def cameraObjects(queue, visibleCorners):

    # Suppress logging for YOLOv8
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    model = YOLO('./yolov8models/yolov8x-visdrone.pt')  # Use a lightweight model for real-time performance

    # Initialize webcam
    cap = cv2.VideoCapture(1)  # 0 is the default webcam

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
    

def resolutionFunction(queue):
    # resolution calibration variables
    confidenceThreshold = 0.1 #varies between 0 and 1
    spatialThreshold = 250 # in transformed coordinates
    ageOff = 60 # in seconds

    # Set up Matplotlib for live plotting
    # define ground translation
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    groundX = 500 # width in cm of ground square
    groundY = 500 # heigh in cm of ground square
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    iconSize = 0.05
    fig, ax = plt.subplots()
    sc = ax.scatter([], [], color='red')
    ax.set_xlim(0, groundX)  # Set limits based on frame size
    ax.set_ylim(0, groundY)
    ax.invert_yaxis()  # Invert y-axis to match OpenCV's coordinate system

    # Initialize tracked objects
    trackedObjects = {} # [id]:[(x,y),TOI, description,identification(friend, enemy, unk), [file_loc]]
    agedTrackedObjects = {}
    objCount = 0
    objTOI = datetime.now()
    #----------------------------------------------------------------------------------------------------------------
    while True:
        if not queue.empty():
            # RESOLVE NEW OBJECTS WITH KNOWN OBJECTS
            source, newObjects, frame = queue.get()
            plotObjects = newObjects.copy()
            # PROCESS IMAGE BASED ENTITIES
            if source == "image":
                for trackedObjId, imEntity in trackedObjects.items():
                    prevX, prevY = imEntity[0]  # position of current tracked entity 

                    # IMAGE OBJECT RESOLUTION
                    for obj_id, (centreX, centreY), label, confidence, bbox in newObjects: # iterate over known list of entities
                        objDist = ((centreX - prevX) ** 2 + (centreY - prevY) ** 2) ** 0.5
                        if objDist < spatialThreshold and confidence > confidenceThreshold:
                            # if the new source is an image, overwrite the old object loc, otherwise leave it alone
                            trackedObjects[trackedObjId][0] = (centreX, centreY) #(x,y)                  
                            trackedObjects[trackedObjId][1] = objTOI #TOI
                            trackedObjects[trackedObjId][2] += f"{source}: {label}" #description,identification(friend, enemy, unk), [file_loc]]
                            newObjects.remove((obj_id, (centreX, centreY), label, confidence, bbox)) # remove item from list if resolved

                
                # ADD NEW OBJECTS
                for obj_id, (centreX, centreY), label, confidence, bbox in newObjects:
                    if confidence > confidenceThreshold:
                        trackedObjects[f"i{objCount}"] = [(centreX, centreY),objTOI,label,None,[None]]
                        objCount += 1                              
                #-------------------------------------------------------------------------------------------------------
                # WEBCAM PLOT
                #-------------------------------------------------------------------------------------------------------

                # Draw bounding boxes, labels, and IDs
                for obj_id, center, label, confidence, bbox in plotObjects:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
                    cv2.putText(frame, f'{label} {confidence:.2f} ID: {obj_id}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Show webcam feed
                cv2.imshow("Webcam Object Tracking", frame)
                
            #-------------------------------------------------------------------------------------------------------
            # AGE OFF TRACKED OBJECTS
            #-------------------------------------------------------------------------------------------------------
            current_time = datetime.now()
            """
            Known issue with difference between currObjTime and current_time being ~20 different despite being called at same time 
            for objKey in trackedObjects.keys():
                currObjTime = trackedObjects[objKey][1]
                print("---------------------------------")
                print(currObjTime)
                print(current_time)
                print((current_time - currObjTime).total_seconds())
            """
            agedTrackedObjects = {key: ent for key, ent in trackedObjects.items() if (current_time - ent[1]).total_seconds() <= ageOff}

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
            
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    

if __name__ == "__main__":      
    import multiprocessing

    groundX = 500 # width in cm of ground square
    groundY = 500 # heigh in cm of ground square
    visibleCorners = [[0,0],[groundX,0],[0,groundY],[groundX,groundY]] #bl, br, tl, tr

    # Create a Queue for inter-process communication
    queue = multiprocessing.Queue()

    # Create and start the processes
    image_process = multiprocessing.Process(target=cameraObjects, args=(queue,visibleCorners,))
    output_process = multiprocessing.Process(target=resolutionFunction, args=(queue,))

    image_process.start()
    output_process.start()

    # Optionally join processes (this example runs indefinitely)
    image_process.join()
    output_process.join()

    cv2.destroyAllWindows()
    