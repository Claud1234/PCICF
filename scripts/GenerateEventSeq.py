import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse
import yaml

def GetSmirkCreateFile(pedestrianAnnotation, offset, numberOfFrames = 100):
    """
    Creates better_smirk_creat.csv for a specific event.
    """

    idOrder = np.array([2, 1, 0, 3, 4, 5])                      # Pedestrian ID order
    pedestrianIDs = np.sort(idOrder[pedestrianAnnotation == 1]) # Translate pedestrian annotation to IDs
    minImagePixel, maxImagePixel = 0, 580
    boxWidth = 60                           # Unit: pixels

    startFrames, endFrames = np.zeros_like(pedestrianIDs), np.zeros_like(pedestrianIDs)
    xStarts, xEnds = np.zeros_like(pedestrianIDs), np.zeros_like(pedestrianIDs)
    yStarts, yEnds = np.zeros_like(pedestrianIDs), np.zeros_like(pedestrianIDs)

    shifts = np.array([offset, (1 + offset), (2 + offset), 0, 1, 2]) * boxWidth      # Unit: pixels
    lastPedestrian = np.argmax([shifts[i] if i in pedestrianIDs else 0 for i in range(shifts.shape[0])])
    
    if lastPedestrian in [0,1,2] :
        startPixel = (minImagePixel - shifts[lastPedestrian])
        pedestrianSpeed = abs(maxImagePixel - startPixel) / (numberOfFrames - 1)        # Unit: pixels/frames
    else:
        startPixel = (maxImagePixel + shifts[lastPedestrian])
        pedestrianSpeed = abs(minImagePixel - startPixel) / (numberOfFrames - 1)        # Unit: pixels/frames

    for i, pedestrianID in enumerate(pedestrianIDs):

        if pedestrianID in [0,1,2]:
            walk = (minImagePixel - shifts[pedestrianID]) + pedestrianSpeed * np.arange(numberOfFrames)
            startFrames[i] = np.argmax(walk>=minImagePixel)

            if shifts[pedestrianID] == shifts[lastPedestrian]:
                endFrames[i] = numberOfFrames - 1
            else:
                endFrames[i] = np.argmax(walk > maxImagePixel) - 1
        else:
            walk = (maxImagePixel + shifts[pedestrianID]) - pedestrianSpeed * np.arange(numberOfFrames)
            startFrames[i] = np.argmax(walk <= maxImagePixel)

            if shifts[pedestrianID] == shifts[lastPedestrian]:
                endFrames[i] = numberOfFrames - 1
            else:
                endFrames[i] = np.argmax(walk < minImagePixel) - 1
        
        xStarts[i], xEnds[i] = int(walk[startFrames[i]]), int(walk[endFrames[i]])
        yStarts[i], yEnds[i] = 200, 200

    dataDict = {"ped_id":pedestrianIDs,"start_x":xStarts, "start_y":yStarts, 
                "end_x":xEnds, "end_y": yEnds, "start_frame":startFrames, "end_frame":endFrames}
    
    df = pd.DataFrame(data=dataDict)
    df.to_csv("better_smirk_creat.csv", index=False)

def GenerateImageSequence(event):
    df = pd.read_csv('./better_smirk_creat.csv')
    print(df)
    evtStr = str(event)

    for i in range(100):
        env_img = cv2.resize(cv2.imread('../datasets/empty_smirk.png'), (640, 480))
        anno_img = np.zeros_like(env_img)
        envimg_file = '%03d' % i + '.png'
        envimg_path = os.path.join('../datasets/MoreSMIRK/raw_data/event_' + evtStr + '/evt_'+evtStr, envimg_file)
        annoimg_path = envimg_path.replace('.png', '.labels.png').replace('evt_' + evtStr, 'evt_' + evtStr + '_anno')
        if not os.path.exists(os.path.dirname(envimg_path)):
            os.makedirs(os.path.dirname(envimg_path))
        if not os.path.exists(os.path.dirname(annoimg_path)):
            os.makedirs(os.path.dirname(annoimg_path))
        cv2.imwrite(envimg_path, env_img)
        cv2.imwrite(annoimg_path, anno_img)

    for idx, row in df.iterrows():
        start_x = row['start_x']
        end_x = row['end_x']
        start_y = row['start_y']
        end_y = row['end_y']
        frame_length = row['end_frame'] - row['start_frame']
        x_interval = abs(row['start_x'] - row['end_x']) / frame_length
        y_interval = abs(row['start_y'] - row['end_y']) / frame_length
        
        if row['ped_id'] in [0,1,2]:                
            seq_path = 'N09pgUrFChEH8GM6APzJ0'      # left to right
        else:
            seq_path = 'aITx4TyRncKnardhTgCzz'      # right to left

        for i in range(frame_length+1):
            print(i)
            rgb_file = 'cam' + '%06d' % (row['start_frame'] + i) + '.png'
            rgb_path = os.path.join('../datasets/smirk', seq_path, rgb_file)
            anno_path = rgb_path.replace('.png', '.labels.png')
            result_rgb_file = '%03d' % (row['start_frame'] + i) + '.png'
            result_rgb_path = os.path.join('../datasets/MoreSMIRK/raw_data/event_' + evtStr + '/evt_' + evtStr, result_rgb_file)
            result_anno_path = result_rgb_path.replace('.png', '.labels.png').replace('evt_' + evtStr, 'evt_' + evtStr + '_anno')
            env_img = cv2.imread(result_rgb_path)
            bgr = cv2.imread(rgb_path)
            bgr = cv2.resize(bgr, (640, 480))
            anno = cv2.imread(anno_path)
            anno = cv2.resize(anno, (640, 480))
            anno_gray = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
            anno_gray = cv2.resize(anno_gray, (640, 480))
            ret, anno_binary = cv2.threshold(anno_gray, 0, 255, cv2.THRESH_BINARY)

            mask_binary = anno_binary == 255
            black_bg_mask = np.zeros_like(bgr)
            black_bg_mask[mask_binary] = bgr[mask_binary]

            pixel_cord = np.argwhere(anno_binary == 255)  # find all pixel coordination for ped contour
            min_y, min_x = np.min(pixel_cord[:], axis=0)
            max_y, max_x = np.max(pixel_cord[:], axis=0)
            height = max_y - min_y
            width = max_x - min_x

            bbox_bgr = bgr[min_y:max_y, min_x:max_x, :]
            bbox_binary = anno_binary[min_y:max_y, min_x:max_x]
            bbox_binary_bool = bbox_binary[:, :] == 255
            bbox_binary_bool_stack = np.stack([bbox_binary_bool, bbox_binary_bool, bbox_binary_bool], axis=2)
            bbox_black_bg = bbox_bgr * bbox_binary_bool_stack
            h, w = bbox_black_bg.shape[0], bbox_black_bg.shape[1]

            if start_x > end_x:
                cord_y = int(start_y - y_interval * i)
                cord_x = int(start_x - x_interval * i)
            else:
                cord_y = int(start_y + y_interval * i)
                cord_x = int(start_x + x_interval * i)
            env_img[cord_y:cord_y+h, cord_x:cord_x+w, :] = env_img[cord_y:cord_y+h, cord_x:cord_x+w, :] * \
                ~bbox_binary_bool_stack + bbox_black_bg

            result_anno = cv2.imread(result_anno_path)
            result_anno[cord_y:cord_y+h, cord_x:cord_x+w, :] = anno[min_y:max_y, min_x:max_x, :]

            cv2.imwrite(result_rgb_path, env_img)
            cv2.imwrite(result_anno_path, result_anno)

def GenerateEvents(firstEvent, lastEvent):
    """
    Retrieves input parameters from MoreSmirkEvents.yml and 
    generates image sequences for the events in the specified 
    range [firstEvent: int(0,103), lastEvent: int(0,103)]. 
    
    """
    fileName = "../datasets/MoreSMIRK/MoreSmirkEvents.yml"

    if not os.path.exists(fileName):
        raise ValueError('The MoreSmirkEvents.yml file is not found.')

    with open(fileName,"r") as file:
        loadedData = yaml.safe_load(file)

    for i in range(firstEvent, lastEvent+1):
        eventDict = loadedData[i]
        eventNumber = eventDict['event']
        activePedestrians = np.array(eventDict['pedestrians'])
        offset = eventDict['offset']
        GetSmirkCreateFile(pedestrianAnnotation=activePedestrians, offset=offset)
        GenerateImageSequence(event=eventNumber)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--single', type=int, help='Single event to generate')
    parser.add_argument('-i', '--interval', type=int, nargs=2, help='Interval of events to generate')
    args = parser.parse_args()
    allowedInput = np.arange(0, 104)

    if args.interval:
        interval = args.interval    # Multiple events to generate
        minEvent, maxEvent = np.min(interval), np.max(interval)
    elif args.single:
        event = args.single         # Single event to generate
        minEvent, maxEvent = event, event
    else:
        raise ValueError('Missing arguments.')
    
    if minEvent in allowedInput and maxEvent in allowedInput:
        GenerateEvents(firstEvent=minEvent, lastEvent=maxEvent)
    else:
        raise ValueError('Events specified are out of range.')