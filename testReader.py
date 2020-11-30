
import numpy as np
import cv2
import h5py

def getData(filename):
    file = h5py.File(filename, 'r')
    print(file.keys())  # to see whats in here
    
    imgs = np.array(file['images'])   
    boxes = np.array(file['labels'])   
    
    file.close()
    
    return imgs, boxes



images, boxes = getData("dataset80.h5")

count = 0
for i in range(0,len(boxes)):
    if sum(boxes[i]) != 0:
        count +=1

print(count)
        
    
    # start= (boxes[i][0],boxes[i][1])
    # end = (boxes[i][2],boxes[i][3])
    # image = cv2.rectangle(images[i], start, end, (0, 0, 255), 2)             
    # cv2.imshow('blah',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
