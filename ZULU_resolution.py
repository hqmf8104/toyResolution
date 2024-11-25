
import cv2
from datetime import datetime, timezone
from toyMaven.plottingFunctions import initialize_plot, update_plot
from toyMaven.cameraFunctions import cameraObjects
from toyMaven.chatFunctions import chatToEntites
import time

def resolutionFunction(queue, gXY):
    # resolution calibration variables
    confidenceThreshold = 0.5 #varies between 0 and 1
    spatialThreshold = 250 # in transformed coordinates
    ageOff = 15 # in seconds

    # Set up Matplotlib for live plotting
    fig, ax = initialize_plot(gXY[0], gXY[1])

    # Map identifiers to image paths, with a default image for unknown types
    image_map = {
        "en": "./icons/en.png",
        "dest": "./icons/dest.png",
    }
    default_image_path = "./icons/unk.png"  # Catch-all image
    
    # Initialize tracked objects
    trackedEntities = {} # [id]:[(x,y),TOI, description,identification(friend, enemy, unk), [file_loc]]
    objCount = 0

    #----------------------------------------------------------------------------------------------------------------
    while True:
        if not queue.empty():
            print("----------------------------------------------------------------------------------------------------------------------------")
            # Get queue
            source, newObjects, frame = queue.get()
            newEntities = {}
            #-------------------------------------------------------------------------------------------------------    
            # PROCESS IMAGE INFORMATION IF REQUIRED
            #-------------------------------------------------------------------------------------------------------
            
            if source == "image":

                # WEBCAM PLOT AND ENTITY STANDARDISATION

                for obj_id, centre, label, confidence, bbox in newObjects:
                    newEntities[f"im_{obj_id}"] = [centre, datetime.now(timezone.utc),f"{source}: {label}","unk",confidence] # standardises image entites for later use
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
                    cv2.putText(frame, f'{label} {confidence:.2f} loc: {centre}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Show webcam feed
                windowName = "Object Tracking - Camera 1" 
                cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(windowName, 600, 523)                              
                #cv2.moveWindow(windowName, 645, 0)
                cv2.imshow(windowName, frame)
                # Move the window to the top-left corner of the screen

            #-------------------------------------------------------------------------------------------------------    
            # PROCESS CHAT INFORMATION IF REQUIRED
            #-------------------------------------------------------------------------------------------------------
            if source == "chat":
                #chat work goes here
                newEntities = newObjects
                [print(f"Chat Object: {key} {ii}") for key, ii in newEntities.items()]

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
                        print(f"Updating {trackedObjId}")
                        # if the new source is an image, overwrite the old object loc, otherwise leave it alone
                        # prioritise more accurate reporting (images)
                        if source == "image":
                            trackedEntities[trackedObjId][0] = xy
                        
                        # update TOI with latest TOI
                        trackedEntities[trackedObjId][1] = TOI

                        # update description with latest description
                        newDesc = f"{source}: {entLabel}"
                        if trackedEntities[trackedObjId][2][:-len(newDesc)] != newDesc: # if the last description added is the same as the new one, do not add a new description
                            trackedEntities[trackedObjId][2] = f"{source}: {entLabel}"
                        
                        # do not overwrite a known identiy (en or dest) with unk
                        if trackedEntities[trackedObjId][3] != "unk" and fofID == "unk":
                            pass
                        else: 
                            trackedEntities[trackedObjId][3] = fofID

                        # update confidence
                        trackedEntities[trackedObjId][4] = conf
                        delIDList.append(obj_id)

                # delete new entities used to update existing entities
                for ii in delIDList:
                    print(f"Deleting: {ii}")
                    del newEntities[ii]
            
            # add new objects
            for obj_id, newEnt in newEntities.items():
                localConf = newEnt[-1]
                if localConf > confidenceThreshold:
                    print(f"Adding {obj_id}_{objCount}")
                    trackedEntities[f"{obj_id}_{objCount}"] = newEnt
                    objCount += 1                              

            # AGE OFF TRACKED OBJECTS
            oldKeys = []
            for key, values in trackedEntities.items():
                print(datetime.now(timezone.utc))
                print(values[1])
                localAge = (datetime.now(timezone.utc) - values[1]).total_seconds()
                print(f"age of {key}: {localAge}")
                if localAge > ageOff:
                    oldKeys.append(key)

            #oldKeys = [key for key, values in trackedEntities.items() if (objTOI - values[1]).total_seconds() > ageOff]
            for key in oldKeys:
                del trackedEntities[key]

            [print(f"Tracked Object: {key} {ii}") for key, ii in trackedEntities.items()]
            #if source == "chat":
            #    time.sleep(5)
                
            #-------------------------------------------------------------------------------------------------------
            # PLOT ENTITIES
            #-------------------------------------------------------------------------------------------------------
            # Update scatter plot with new points. Note it will error out if there are no objects detected without if statement
            update_plot(ax, trackedEntities, image_map, default_image_path)

                
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
if __name__ == "__main__":      
    import multiprocessing
    # define ground conditions
    groundX = 500 # width in cm of ground square
    groundY = 500 # heigh in cm of ground square
    visibleCorners = [[0,0],[groundX,0],[0,groundY],[groundX,groundY]] #bl, br, tl, tr
    

    # Create a Queue for inter-process communication
    queue = multiprocessing.Queue()

    # Create and start the processes
    image_process = multiprocessing.Process(target=cameraObjects, args=(queue,visibleCorners,))
    text_process = multiprocessing.Process(target=chatToEntites, args=(queue,))
    output_process = multiprocessing.Process(target=resolutionFunction, args=(queue,(groundX,groundY),))

    image_process.start()
    text_process.start()
    output_process.start()

    # Optionally join processes (this example runs indefinitely)
    image_process.join()
    output_process.join()

    cv2.destroyAllWindows()
    