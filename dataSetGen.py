# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:16:46 2020

@author: scm29
"""
""

import numpy as np
import cv2
import os
import pandas as pd
import xml.etree.ElementTree as et
import math
import h5py


def get_bb(event, col, row, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if param[1] == 0:
            param[0][0] = col   # col = x_coordinate
            param[0][1] = row   # row = y_coordinate
            param[1] += 1
        elif param[1] == 1:
            param[0][2] = col
            param[0][3] = row
            param[1] = 2

def getIndices(dframe, imFile):
    for i in range(len(dframe)):
        if dframe.at[i,"image"] == img:
            return i
    return -1

# going to return a 128x128 image if possible
def standardize(image, mins, maxes):
    
    m = len(image)
    n = len(image[0])
    midx = math.floor((maxes[0]+mins[0])/2)
    midy = math.floor((maxes[1]+mins[1])/2)
    
    xmin = max(0,midx-64)
    xmax = min(n,midx+64)
    ymin = max(0,midy-64)
    ymax = min(m,midy+64)
    
    newimg = image[ymin:ymax,xmin:xmax]
    
    return newimg,len(newimg),len(newimg[0])


dic = {"image": [],"Dimensions": []}
for i in range(1,6):
	dic[f'Object {i}']=[]

# gather samples 
for file in os.listdir("./annotations"):
    row = []
    xml = et.parse("./annotations/"+file) 
    root = xml.getroot()
    img = root[1].text
    row.append(img)
    h,w = root[2][0].text,root[2][1].text
    row.append([h,w])
    
    for i in range(4,len(root)):
        temp = []
        temp.append(root[i][0].text)
        for point in root[i][5]:
            temp.append(point.text)
        row.append(temp)
    for i in range(len(row),10):
        row.append(0)
    for i,each in enumerate(dic):
        dic[each].append(row[i])
        # print(row[i])
df = pd.DataFrame(dic)
# print(df.head())# prints first 5rows in the data frame
# print(df.at[0,'Object 2']) # prints an 'Object' for reference

# lets take a sub sample of these images
# goal is to have 40 masked and 40 unmaked images
labels = np.zeros(shape = (80, 4), dtype='uint8')   # will resize after
images = np.zeros(shape = (80, 128, 128, 3), dtype='uint8') 
masked = 0
unmasked = 0
index = 0
for img in os.listdir("./images"):
    
    if masked >= 40 and unmasked >= 40:   # max samples of each
        break
    
    dfInd = getIndices(df, img) # if in different orders
    image = cv2.imread("./images/" + img)
    
    for i in range(1,6):
            
        if df.at[dfInd, "Object "+str(i)] != 0:
            
            start= (int(df.at[dfInd, "Object "+str(i)][1]),int(df.at[dfInd, "Object "+str(i)][2]))
            end = (int(df.at[dfInd, "Object "+str(i)][3]),int(df.at[dfInd, "Object "+str(i)][4]))
            # image = cv2.rectangle(image, start, end, (0, 0, 255), 2)             
            # cv2.imshow('blah',image)
            # cv2.destroyAllWindows()
            dataImage,m,n = standardize(image, start, end)
            
            if m == 128 and n == 128:
                if df.at[dfInd, "Object "+str(i)][0] == "with_mask":
                    if masked < 40:
                        print('IS wearing a mask')
                        cv2.namedWindow('INPUT')
                        bb = [0,0,0,0]
                        points = 0
                        cv2.setMouseCallback('INPUT', get_bb, [bb,points])
                        cv2.imshow('INPUT',dataImage)
                        k = cv2.waitKey(0)
                        if k == 27:         # wait for ESC key to exit
                            cv2.destroyAllWindows()
                        elif k == ord('s'): # wait for 's' key to save and exit
                            print(bb)
                            labels[index] = bb
                            images[index] = dataImage
                            cv2.destroyAllWindows() 
                            masked += 1
                            index +=1
                            print(index)
                elif df.at[dfInd, "Object "+str(i)][0] == "without_mask":
                    if unmasked < 40:
                        print('not wearing a mask')
                        cv2.imshow('INPUT',dataImage)
                        k = cv2.waitKey(0)
                        if k == 27:      # wait for ESC or close to exit
                            cv2.destroyAllWindows()
                        elif k == ord('s'): # wait for 's' key to save and exit
                            cv2.destroyAllWindows() 
                            labels[index] = [0,0,0,0]
                            images[index] = dataImage
                            unmasked += 1
                            index+=1
                            print(index)
print(masked, unmasked)

hf = h5py.File('dataset80.h5', 'w')
hf.create_dataset('images', data=images)
hf.create_dataset('labels', data=labels)
hf.close()

# def calcPrecision(oldBoxes, newBoxes):
#     p = []
    
#     """label_start= (boxes[ii][1],boxes[ii][0])
#     label_end = (boxes[ii][3],boxes[ii][2])
#     """
    
#     for i in range(len(newBoxes)):
        
#         """Ri = MAX[0, MIN(AX2, BX2) - MAX(AX1, BX1)] * MAX[0, MIN(AY2, BY2) - MAX(AY1, BY1)]"""
#         ri = max(0,min(oldBoxes[i][3], newBoxes[i][2])-max(oldBoxes[i][1], newBoxes[i][0]))*max(0,min(oldBoxes[i][2], newBoxes[i][3])-max(oldBoxes[i][0], newBoxes[i][1]))
        
#         """Then the union can be found:
#         Ru = Ra + Rb - Ri"""
#         ra = int((oldBoxes[i][3]-oldBoxes[i][1])*(oldBoxes[i][2]-oldBoxes[i][0]))
#         rb = int((newBoxes[i][3]-newBoxes[i][1])*(newBoxes[i][2]-newBoxes[i][0]))
#         ru = ra+rb-ri
        
#         """Then the percentage of overlap can be calculated:
#         % Overlap = Ri / Ru"""

#         p.append(ri/ru)
    
#     return p


# if __name__ == '__main__':
#     images, boxes = getData("datasetNew.h5")
    
#     #classes = []
#     newLabels = []
#     oldLabels = []
#     outputs = np.zeros(shape = (100, 128, 128, 3), dtype = 'uint8')
    
#     for ii in range(10):
#         im = np.zeros(shape = (128,128,3), dtype = 'uint8')
#         image = im+images[ii]
#         bb_i = getBoundBox(image, False)
#         maskFound, bb = filterBoundingBoxes(bb_i, 0.3)
#         #classes.append(maskFound)
#         #print(ii, maskFound)
        
#         # print(bb)
#         # dot = cv2.circle(images[20], (bb['x'],bb['y']), radius=3, color=(255, 0, 0), thickness=-1)
#         # dot = cv2.circle(dot, (bb['x']+bb['w'],bb['y']+bb['h']), radius=3, color=(255, 0, 0), thickness=-1)
#         og = np.zeros(shape = (128,128,3), dtype = 'uint8') + images[ii]
#         if boxes[ii].all() != 0:  
#             oldLabels.append([boxes[ii][1],boxes[ii][0],boxes[ii][3],boxes[ii][2]])
#             label_start= (boxes[ii][1],boxes[ii][0])
#             label_end = (boxes[ii][3],boxes[ii][2])
#             og = cv2.rectangle(images[ii], label_start, label_end, (0, 255, 0), 2) 
#         else:
#             oldLabels.append([boxes[ii][1],boxes[ii][0],boxes[ii][3],boxes[ii][2]])
        
#         if maskFound:
#             newLabels.append([bb['x'], bb['y'], bb['x'] + bb['w'], bb['y'] + bb['h']])
#             og = cv2.rectangle(og, (bb['x'], bb['y']), (bb['x'] + bb['w'], bb['y'] + bb['h']), (0, 0, 255), 2)
#         else:
#             newLabels.append([bb['x'], bb['y'], bb['x'] + bb['w'], bb['y'] + bb['h']])
            
            
#         np.append(outputs, og)
            
#         cv2.imshow('with filtered box',og)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
    
#     # print(classes)
#     # print(oldLabels)
#     # print(newLabels)
#     precision = calcPrecision(oldLabels, newLabels)
    
#     print(precision)
