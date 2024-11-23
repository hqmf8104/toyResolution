
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import deque
from toyMaven.spatialFunctions import F1
from datetime import datetime, timedelta
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import logging
from toyMaven.plottingFunctions import getImage

def cameraObjects(queue, visibleCorners, cameraChoice = 1):

    # Suppress logging for YOLOv8
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    model = YOLO('./yolov8models/yolov8x-visdrone.pt')  # Use a lightweight model for real-time performance

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
    
def resolutionFunction(queue):
    # resolution calibration variables
    confidenceThreshold = 0.5 #varies between 0 and 1
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
    trackedEntities = {} # [id]:[(x,y),TOI, description,identification(friend, enemy, unk), [file_loc]]
    agedtrackedEntities = {}
    objCount = 0
    objTOI = datetime.now()
    #----------------------------------------------------------------------------------------------------------------
    while True:
        if not queue.empty():
            # Get queue
            source, newObjects, frame = queue.get()
            newEntities = {}
            #-------------------------------------------------------------------------------------------------------    
            # PROCESS IMAGE INFORMATION IF REQUIRED
            #-------------------------------------------------------------------------------------------------------
            
            if source == "image":

                # WEBCAM PLOT AND ENTITY STANDARDISATION               
                for obj_id, centre, label, confidence, bbox in newObjects:
                    newEntities[obj_id] = [centre,objTOI,f"{source}: {label}","unk",confidence] # standardises image entites for later use
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
                    cv2.putText(frame, f'{label} {confidence:.2f} loc: {centre}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Show webcam feed
                cv2.imshow("Webcam Object Tracking", frame)

                    
            #-------------------------------------------------------------------------------------------------------    
            # PROCESS CHAT INFORMATION IF REQUIRED
            #-------------------------------------------------------------------------------------------------------
            if source == "chat":
                #chat work goes here
                pass

            #-------------------------------------------------------------------------------------------------------
            # RESOLVE ENTITIES
            #-------------------------------------------------------------------------------------------------------
            # COMPARE NEW ENTITIES TO TRACKED ENTITIES
            for trackedObjId, imEntity in trackedEntities.items():
                prevX, prevY = imEntity[0]  # position of current tracked entity 

                # check existing objects
                delIDList = []
                for obj_id, [xy,TOI,entLabel,fofID,conf] in newEntities.items(): # iterate over known list of entities
                    objDist = ((xy[0] - prevX) ** 2 + (xy[1] - prevY) ** 2) ** 0.5
                    if conf > confidenceThreshold and objDist < spatialThreshold:
                        # if the new source is an image, overwrite the old object loc, otherwise leave it alone
                        trackedEntities[trackedObjId][:2] = [xy,TOI]
                        newDesc = f"{source}: {entLabel}"
                        if trackedEntities[trackedObjId][2][:-len(newDesc)] != newDesc: # if the last description added is the same as the new one, do not add a new description
                            trackedEntities[trackedObjId][2] = f"{source}: {entLabel}"
                        trackedEntities[trackedObjId][3:] = fofID, conf
                        delIDList.append(obj_id)
                for ii in delIDList:
                    del newEntities[ii]
            
            # add new objects
            for obj_id, newEnt in newEntities.items():
                conf = newEnt[-1]
                if conf > confidenceThreshold:
                    trackedEntities[f"{obj_id}_{objCount}"] = newEnt
                    objCount += 1                              

            # AGE OFF TRACKED OBJECTS
            current_time = datetime.now()
            agedtrackedEntities = {key: ent for key, ent in trackedEntities.items() if (current_time - ent[1]).total_seconds() <= ageOff}
            [print(f"{key}: {ii}") for key, ii in agedtrackedEntities.items()]
            #-------------------------------------------------------------------------------------------------------
            # PLOT ENTITIES
            #-------------------------------------------------------------------------------------------------------
            # Update scatter plot with new points. Note it will error out if there are no objects detected without if statement
            import matplotlib.image as mpimg
                        
            enIcon = mpimg.imread("./icons/en.png")
            unkIcon = mpimg.imread("./icons/unk.png")

            # clear scatter plot
            for artist in ax.artists:  # ax.artists contains all added AnnotationBbox objects
                artist.remove()
                
            # go through each object and plot
            for ent in agedtrackedEntities.values():

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
                tracked_object_count = len(agedtrackedEntities)
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
    