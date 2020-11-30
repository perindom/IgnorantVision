import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
from GaussianBlur import gaussian_blur
from EdgeDetection import sobel


def getData(filename):
    file = h5py.File(filename, 'r')
    # print(file.keys())  # to see whats in here

    imgs = np.array(file['images'])
    boxes = np.array(file['labels'])

    file.close()

    return imgs, boxes


def getBoundBox(image, displaySteps):
    # Input image and convert to grayscale
    # Apply Gaussian Blur to smooth over hard intensity deltas
    # Use sobel filter to get edge detection image
    # convert sobel to a binary image with threshold image
    # Using the built in opencv findContours function, find contours from the binary image
    # get the biggest three contours and return them and their information

    first_c = image
    if displaySteps:
        cv2.imshow('noblur', first_c)
        cv2.waitKey(0)

    # Applied a Grayscale filter
    first = cv2.cvtColor(first_c, cv2.COLOR_RGB2GRAY)

    if displaySteps:
        cv2.imshow('bw', first)
        cv2.waitKey(0)

    # Apply Gaussian Blur to get rid of any insignificant hard edges
    first_b = gaussian_blur(first, 17, verbose=displaySteps)

    if displaySteps:
        plt.imshow(first_b, cmap='gray')
        plt.title('Gaussian Blur'), plt.xticks([]), plt.yticks([])
        plt.show()

    # Use sobel filter edge detection
    sobel_b = sobel(first_b, 3)

    if displaySteps:
        plt.imshow(sobel_b, cmap='gray')
        plt.title('Sobel Edge Detection'), plt.xticks([]), plt.yticks([])
        plt.show()

    sbi = np.uint8(sobel_b)
    sbinary = cv2.threshold(sbi, 200, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

    contours, hierarchy = cv2.findContours(sbinary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if displaySteps:
        cv2.imshow('Sobel Edges Binary Contoured', sbinary)
        cv2.waitKey(0)

    # Find the index of the largest contours
    areas = [cv2.contourArea(c) for c in contours]

    largest_area_ind = sorted(range(len(areas)), key=lambda x: areas[x])[-3:]
    cnt = [contours[i] for i in largest_area_ind]
    bbs = []

    for c in cnt:
        x, y, w, h = cv2.boundingRect(c)
        first_bounded = cv2.rectangle(first_c, (x, y), (x + w, y + h), (0, 0, 255), 2)
        bb = {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "contour": c,
            "binary_crop": sbinary[x:x + w, y:y + h]
        }
        bbs.append(bb)

    # Make sure there are at least 3 bounding boxes, if not, add duplicates of the last bb found
    while len(bbs) < 3:
        bbs.append(bb)

    bb_dict = {
        "bb1": bbs[0],
        "bb2": bbs[1],
        "bb3": bbs[2],
        "first_bounded": first_bounded,
        "first_c": first_c
    }
    return bb_dict


def filterBoundingBoxes(bbs, intensity_threshold):
    # Each Image has three bounding boxes stored in bb_dict
    # Each bounding box has a binary crop of what that box is "bounding"
    # Check each box, whichever has a lower proportion of light values is likely to have the mask
    # return that bounding box
    just_bbs = [bbs["bb1"], bbs["bb2"], bbs["bb3"]]
    intensities = []
    for bb in just_bbs:
        binary = bb["binary_crop"]
        total_intensity = np.sum(binary) / 255
        size = binary.shape[0] * binary.shape[1]
        intensity_proportion = total_intensity / size
        intensities.append(intensity_proportion)

    best_int_ind = intensities.index(max(intensities))
    # return a boolean and the lowest intensity bounding box
    return (intensities[best_int_ind] > intensity_threshold), just_bbs[best_int_ind]


# compare label/s from data set to labels found in script
# return accuracy of bounding box on 0-1 scale
def calcAccuracy(pred, test):
    valid = []
    for t_box in test:
        valid.append(not (all(t == 0 for t in t_box)))
    return sum(x == y for x, y in zip(pred, valid)) / len(pred)


def calcPrecision(newBoxes, oldBoxes, classes):
    p = []
    for i in range(len(newBoxes)):
        if oldBoxes[i].all() != 0:
            # A is the original, oldbox
            AX1 = oldBoxes[i][0]
            AY1 = oldBoxes[i][1]
            AX2 = oldBoxes[i][2]
            AY2 = oldBoxes[i][3]
            # B is the newbox from our filter
            BX1 = newBoxes[i][0]
            BY1 = newBoxes[i][1]
            BX2 = newBoxes[i][2]
            BY2 = newBoxes[i][3]

            ra = (AY2 - AY1 + 1) * (AX2 - AX1 + 1)
            rb = (BY2 - BY1 + 1) * (BX2 - BX1 + 1)

            """Ri = MAX[0, MIN(AX2, BX2) - MAX(AX1, BX1)] * MAX[0, MIN(AY2, BY2) - MAX(AY1, BY1)]"""
            ri = max(0, (min(AX2, BX2) - max(AX1, BX1) + 1)) * max(0, (min(AY2, BY2) - max(AY1, BY1) + 1))

            """Then the union can be found: Ru = Ra + Rb - Ri"""
            ru = ra + rb - ri

            """Then the percentage of overlap can be calculated: % Overlap = Ri / Ru"""
            prec = ri / ru
            p.append([prec, i])
    return p





def run():
    count = 0
    images, boxes = getData("/Users/dominickperini/Documents/GitHub/IgnorantVision/dataSet/dataset80.h5")

    outputs = []
    binary_classifications = []
    bbs = []

    for ii in range(len(images)):
        im = np.zeros(shape=(128, 128, 3), dtype='uint8')
        image = im + images[ii]

        # Get three bounding boxes for the image
        bb_i = getBoundBox(image, False)

        # Look at the bounding boxes and determine if there is a mask, and which bb has it
        bin_class, bb = filterBoundingBoxes(bb_i, 0.38)

        binary_classifications.append(bin_class)

        bounded = np.zeros(shape=(128, 128, 3), dtype='uint8') + images[ii]
        if boxes[ii].all() != 0:
            count += 1
            label_start = (boxes[ii][0], boxes[ii][1])
            label_end = (boxes[ii][2], boxes[ii][3])
            bounded = cv2.rectangle(images[ii], label_start, label_end, (0, 255, 0), 2)

        bounded = cv2.rectangle(bounded, (bb['x'], bb['y']), (bb['x'] + bb['w'], bb['y'] + bb['h']), (0, 0, 255), 2)
        bbs.append([bb['x'], bb['y'], bb['x'] + bb['w'], bb['y'] + bb['h']])

        outputs.append(bounded)

    hstacks = []
    mosaic = []

    for i in range(len(images) // 10):
        hstacks.append(cv2.hconcat(outputs[i * 10: 10 + i * 10]))

    mosaic = cv2.vconcat(hstacks)
    accuracy = calcAccuracy(binary_classifications, boxes)
    prec = calcPrecision(bbs, boxes, binary_classifications)
    prec = sorted(prec, reverse=True)
    precs = [row[0] for row in prec]
    indexes = [row[1] for row in prec]

    top10 = []
    for i in indexes[0:10]:
        top10.append(outputs[i])

    top10mosaic = cv2.hconcat(top10)
    cv2.imshow('Top 10 Precision Images', top10mosaic)
    cv2.waitKey(0)
    print("Top 10 Precisions Average:", sum(precs[0:10]) / len(precs[0:10]))
    print("Accuracy:", accuracy)
    print("Average Precision of Masked images:", sum(precs) / len(precs))

    cv2.imshow('Output Mosaic of ignorant.py', mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    run()