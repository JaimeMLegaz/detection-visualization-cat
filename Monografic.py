import fnmatch
import sys
import easygui as easygui
import numpy as np
import cv2 as cv
import pydicom as dicom
import glob
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
import os
import math
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d


# FILTERING FUNCTIONS
def filter_contours_lungs(contours, hierarchy):  # Receives contours and return those assigned to the lungs
    to_delete = []

    for i in range(0, len(hierarchy[0]),1):  # We choose the exterior contours (without the outermost one)
        if hierarchy[0, i, 3] == -1:
            to_delete.append(i)

    i = 0

    for cnt in contours:
        if len(cnt) < 120: # We delete those contours too small to be a lung
            to_delete.append(i)
        i = i + 1

    cont = np.delete(contours, to_delete, axis=0)

    return cont


def filter_contours_artery(contours, hierarchy, last_img):  # Receives contours, returns associated with the artery
    to_return = []
    to_delete = []
    for i in range(0, len(hierarchy[0]), 1):  # We chose the outer contours (without the outermost one)
        if hierarchy[0, i, 3] == -1:
            to_delete.append(i)
    cont = np.delete(contours, to_delete, axis=0)

    for cnt in cont: # Among the interior contours
        center, radio = cv.minEnclosingCircle(cnt)
        round = cv.contourArea(cnt) / (np.pi * radio ** 2)
        if (((cv.contourArea(cnt) < 1000) and (cv.contourArea(cnt) > 100)) and (round > 0.8)) or (overlap(cnt, last_img)):
            to_return.append(cnt)

    return to_return


def overlap(cnt, last_img):  # TODO Esto qué hace?
    new_img = np.zeros(last_img.shape)
    cv.drawContours(new_img, [cnt], 0, (255, 255, 255), -1)
    new_img2 = new_img * last_img

    return cv.norm(new_img2, cv.NORM_L1) > 0


def filter_contours_bronchi(contours, hierarchy):  # Receives contours, returns those associated with the bronchi
    to_return = []

    for cnt in contours:
        if cv.contourArea(cnt) > 10: # We delete those contours that are too small to be a lung
            to_return.append(cnt)

    return to_return


#### DETECTING FUNCTION ####

def show_img(img, last_img, mode=0, erode=2, dilate=2):

    cont_lung = np.empty([1, 2])  # Initializes the matrices that will contain the lungs and/or bronchi
    cont_bronchi = None  # This will contain the contours of the bronchi
    cont_art = None  # This will contain the contours of the artery
    next_img = np.zeros(img.shape)  # This is for the mask of the "Mode 4"

    if mode == 0:  # Thresholding + Contour detection
        # cv.imshow("pre_thresh",img)
        ret, th2 = cv.threshold(img, 30, 255, cv.THRESH_BINARY)  # Thresholding
        contours, hierarchy = cv.findContours(th2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # Contour selection

        cont_lung = filter_contours_lungs(contours, hierarchy)  # We obtain the contours of the lungs

        cv_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # th2 -> view the threshold image, img -> view the original img

        cv.drawContours(cv_img, cont_lung, -1, (0, 255, 0), 2)  # We draw the contours over the original image
        # cv.imshow("post_thresh", th2)
        cv.imshow("window", cv_img)

    elif mode == 1:  # Thresholding + Contours detection + Erosion + Dilatation
        ret, th2 = cv.threshold(img, 30, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)  # Kernel for dilatation/erosion
        th2 = cv.erode(th2, kernel, iterations=erode)  # Erosion
        th2 = cv.dilate(th2, kernel, iterations=dilate)  # Dilatation
        contours, hierarchy = cv.findContours(th2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.imshow("original", img)
        cont_lung = filter_contours_lungs(contours, hierarchy)

        cv_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # th2 -> view the threshold image, img -> view the original img

        cv.drawContours(cv_img, cont_lung, -1, (0, 255, 0), 2)
        cv.imshow("window", cv_img)

    elif mode == 2:  # Thresholding + Contours detection + Erosion + Dilatation + Artery detection

        # Step 1. Artery

        ret, th_art = cv.threshold(img, 5, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        th_art = cv.morphologyEx(th_art, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # Removal of noise
        img_artery = th_art[190:400, 175:325]  # We trim the image to get the area where the artery tends to be
        img_artery = cv.erode(img_artery, kernel)  # We want to expand the artery, in order to detect it better
        contours_art, hierarchy_art = cv.findContours(img_artery, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        cont_art = [cnt + (175, 190) for cnt in contours_art]  # To compensate for the previous trim
        cont_art = filter_contours_artery(cont_art, hierarchy_art, last_img)  # We get the artery contours
        # At this point, we have successfully obtained the artery's contours

        # Step 2. Lungs

        ret, th_lung = cv.threshold(img, 30, 255, cv.THRESH_BINARY)

        # cv.imshow("preborrado_arteria",th_lung)

        cv.fillPoly(th_lung, pts=cont_art, color=(255, 255, 255))  # We color in white what corresponds to the artery

        # cv.imshow("postborrado_arteria", th_lung)

        th_lung = cv.erode(th_lung, kernel, iterations=erode)  # Erosion and dilatation of the lungs
        th_lung = cv.dilate(th_lung, kernel, iterations=dilate)
        contours_lung, hierarchy_lung = cv.findContours(th_lung, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cont_lung = filter_contours_lungs(contours_lung, hierarchy_lung)

        # Step 3. Draw the contours

        cv_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # th2 -> view the threshold image, img -> view the original img

        cv.drawContours(cv_img, cont_art, -1, (0, 0, 255), 2)  # Artery contours (red)
        cv.drawContours(next_img, cont_art, -1, (255, 255, 255), -1)  # We add the artery to the current image (white)
        cv.drawContours(cv_img, cont_lung, -1, (0, 255, 0), 2)  # Lung contours (green)
        cv.imshow("window", cv_img)

    elif mode == 3:  # Thresholding + Contours detection + Erosion + Dilatation + Artery detection + Bronchi detection

        # Step 1. Artery

        ret, th_art = cv.threshold(img, 5, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        th_art = cv.morphologyEx(th_art, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))  # Noise removal
        img_artery = th_art[190:400, 175:325]  # We trim the image to get the area where the artery tends to be
        contours_art, hierarchy_art = cv.findContours(img_artery, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        cont_art = [cnt + (175, 190) for cnt in contours_art]  # To compensate for the previous trim
        cont_art = filter_contours_artery(cont_art, hierarchy_art, last_img)  # We get the artery contours
        # At this point, we have successfully obtained the artery's contours

        # Step 2. Lungs

        ret, th_lung = cv.threshold(img, 30, 255, cv.THRESH_BINARY)
        cv.fillPoly(th_lung, pts=cont_art, color=(255, 255, 255))  # We color in white what corresponds to the artery
        th_lung = cv.erode(th_lung, kernel, iterations=erode)  # Erosion and dilatation of the lungs
        th_lung = cv.dilate(th_lung, kernel, iterations=dilate)
        contours_lung, hierarchy_lung = cv.findContours(th_lung, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cont_lung = filter_contours_lungs(contours_lung, hierarchy_lung)

        # Step 3. Bronchi

        # 3.1. Draw the contours, since we want to draw the bronchi over the current contours
        cv_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # th2 -> view the threshold image, img -> view the original img

        cv.drawContours(cv_img, cont_art, -1, (0, 0, 255), 2)
        cv.drawContours(next_img, cont_art, -1, (255, 255, 255), -1)  # We add the artery to the current image (white)
        cv.drawContours(cv_img, cont_lung, -1, (0, 255, 0), 2)

        count = 1  # For the window name of each lung

        # 3.2. Finding the bronchi and drawing them
        if cont_lung.size > 0:
            for c in cont_lung:  # For each lung
                max_izq = tuple(c[c[:, :, 0].argmin()][0])[0]  # We obtain the outermost coordinates for that lung...
                max_der = tuple(c[c[:, :, 0].argmax()][0])[0]  # ...since we will only select bronchi inside it
                max_top = tuple(c[c[:, :, 1].argmin()][0])[1]
                max_bot = tuple(c[c[:, :, 1].argmax()][0])[1]

                img_bronchi = img[max_top:max_bot, max_izq:max_der]  # We select the image from inside the lung

                ret, th_bronchi = cv.threshold(img_bronchi, 12, 255, cv.THRESH_BINARY_INV)  # Arbitrary threshold of 12, works well in 2/3 image sets

                # cv.imshow("wind" + str(count), th_bronchi)

                contours_bronchi, hierarchy_bronchi = cv.findContours(th_bronchi, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                cont_bronchi = filter_contours_bronchi(contours_bronchi, hierarchy_bronchi)
                cont_bronchi = [cnt + (max_izq, max_top) for cnt in cont_bronchi]  # To compensate for previous slice
                cv.drawContours(cv_img, cont_bronchi, -1, (255, 0, 0), 1)  # Drawing the contours of the bronchi found

                count = count + 1  # For the window name of each lung

        cv.imshow("window", cv_img)

    elif mode == 4:  # Thresholding + Contours + Erosion + Dilatation + Artery detection + Adaptive bronchi detection

        # Step 1. Artery

        ret, th_art = cv.threshold(img, 5, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        th_art = cv.morphologyEx(th_art, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))  # Noise removal
        img_artery = th_art[190:400, 175:325]  # We trim the image to get the area where the artery tends to be
        contours_art, hierarchy_art = cv.findContours(img_artery, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        cont_art = [cnt + (175, 190) for cnt in contours_art]  # To compensate for the previous slice
        cont_art = filter_contours_artery(cont_art, hierarchy_art, last_img)  # We get the artery contours
        # At this point, we have successfully obtained the artery's contours

        # Step 2. Lungs

        ret, th_lung = cv.threshold(img, 30, 255, cv.THRESH_BINARY)
        cv.fillPoly(th_lung, pts = cont_art, color=(255, 255, 255))  # We color in white what corresponds to the artery
        th_lung = cv.erode(th_lung, kernel, iterations=erode)  # Erosion and dilatation of the lungs
        th_lung = cv.dilate(th_lung, kernel, iterations=dilate)
        contours_lung, hierarchy_lung = cv.findContours(th_lung, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cont_lung = filter_contours_lungs(contours_lung, hierarchy_lung)

        # Step 3. Bronchi

        # 3.1. Draw the contours, since we want to draw the bronchi over the current contours
        cv_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # th2 -> view the threshold image, img -> view the original img

        cv.drawContours(cv_img, cont_art, -1, (0, 0, 255), 2)
        cv.drawContours(next_img, cont_art, -1, (255, 255, 255), -1)  # We add the artery to the current image (white)
        cv.drawContours(cv_img, cont_lung, -1, (0, 255, 0), 2)

        # 3.2. Obtain the mask for the lungs and their norm
        mask = np.zeros((512, 512), np.uint8)  # Base image for the mask

        cv.fillPoly(mask, pts=cont_lung, color=(255, 255, 255))  # We color the lungs in white, to form the mask
        # cv.imshow("mask", mask)

        norm_mask = cv.norm(mask, cv.NORM_L1)  # We get the result 255*n, since n is the number of white pixels

        # 3.3. Obtain the norm of the image covered by the mask
        norm_img = cv.norm(img, cv.NORM_L1, mask) # Obtain the sum of the grey values of the lungs

        # 3.4. Obtain the mean grey value of inside the lungs
        if norm_mask > 0.0:  # We have to check if there are any visible lungs
            grey_lungs = float(norm_img) / (float(norm_mask) / 255)  # grey_lungs = mean grey value

            # 3.2. Find and draw the bronchi
            if cont_lung.size > 0:
                for c in cont_lung:  # For each lung
                    max_izq = tuple(c[c[:, :, 0].argmin()][0])[0]  # We obtain the outermost coordinates for that lung
                    max_der = tuple(c[c[:, :, 0].argmax()][0])[0]  # ...since we will only select bronchi inside it
                    max_top = tuple(c[c[:, :, 1].argmin()][0])[1]
                    max_bot = tuple(c[c[:, :, 1].argmax()][0])[1]

                    img_bronchi_0 = img  # We copy the original image
                    # cv.imshow("img_bronchi_0",img_bronchi_0)
                    # cv.imshow("mask",(255 - mask))
                    img_bronchi_0 = cv.add(img_bronchi_0, (255 - mask))  # Remove from this image what is not a lung

                    # cv.imshow("result", img_bronchi_0)
                    img_bronchi = img_bronchi_0[max_top:max_bot, max_izq:max_der]  # Select only from inside the lungs

                    ret, th_bronchi = cv.threshold(img_bronchi, grey_lungs * 1.5, 255, cv.THRESH_BINARY_INV)  # th was 12
                    # cv.imshow("bronq", th_bronchi)
                    contours_bronchi, hierarchy_bronchi = cv.findContours(th_bronchi, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    cont_bronchi = filter_contours_bronchi(contours_bronchi, hierarchy_bronchi)
                    cont_bronchi = [cnt + (max_izq, max_top) for cnt in cont_bronchi]  # Compensate for previous slice
                    cv.drawContours(cv_img, cont_bronchi, -1, (255, 0, 0), 1)  # Draw the contours of the bronchi

        cv.imshow("window", cv_img)
    cv.waitKey(0)
    return cont_lung, cont_art, cont_bronchi, next_img


################## INITIAL VARIABLES #######################

PATH = ["SIB Imagenes seleccionadas/Case 1/",
        "SIB Imagenes seleccionadas/Case 2/",
        "SIB Imagenes seleccionadas/Case 3/"]  # List of the paths to the three image sets

# Index that determines which set are we working with
# imageSetNumber = int(easygui.buttonbox("Select a image set:", "Select image set", ("0", "1", "2")))
mode = int(easygui.buttonbox("Select a mode:\n"
                             "0: Thresholding + Contours detection\n"
                             "1: Mode 0 + Erosion and dilatation\n"
                             "2: Mode 1 + Artery detection\n"
                             "3: Mode 2 + Bronchi detection\n"
                             "4: Mode 2 + Adaptive bronchi detection",
                             "Select mode set", ("0", "1", "2", "3", "4")))  # Index of the mode we're working at

# Here we select the images directory
chosen_dir = easygui.diropenbox("Choose the directory that contains the .dcm files")
if len(fnmatch.filter(os.listdir(chosen_dir), '*.dcm')) == 0:
    easygui.msgbox("ERROR: Selected directory contains no .dcm files")
    sys.exit(1)

image_files = glob.glob(chosen_dir+"/*.dcm")
image_files.sort()
imageSet = [(dicom.dcmread(img)).pixel_array for img in image_files]  # List containing the image sets

################### NORMALIZATION ######################

maxGrey = 0
minGrey = 9999999  # Just needs to be higher than 65536

for i in range(0, len(imageSet)):  # Maximum and minimum grey values
    greyValue = np.amax(imageSet[i])
    if greyValue > maxGrey:
        maxGrey = greyValue

    greyValue = np.amin(imageSet[i])

    if greyValue < minGrey:
        minGrey = greyValue


factor = 255.0/(maxGrey - minGrey)
for i in range(0, len(imageSet)):  # Normalizing
    imageSet[i] = ((imageSet[i] - minGrey) * factor).astype(np.uint8)


################# OBTAINING THE CONTOURS #####################

# Set 1 (Lungs)
X1 = []
Y1 = []
Z1 = []
# Set 2 (Bronchi)
X2 = []
Y2 = []
Z2 = []
# Set 3 (Artery)
X3 = []
Y3 = []
Z3 = []

last_img = np.zeros(imageSet[0].shape)

for i in range(0, len(imageSet)):
    print(i)  # To know in which image we are at TODO: Dibujarlo en la imagen
    cont_lung, cont_art, cont_bronchi, next_img = show_img(imageSet[i], last_img, mode)  # We show the image AND obtain
    # the contours

    last_img = next_img
    for lung in cont_lung:  # For each lung
        for point in lung:  # For each point of the contour
            X1.append(point[0, 0])  # We obtain its X coordinate
            Y1.append(point[0, 1])  # We obtain its Y coordinate
            Z1.append((len(imageSet) - i))  # We establish its Z coordinate, which depends on the image index
    if cont_bronchi is not None:
        for bronchus in cont_bronchi:
            for point in bronchus:
                X2.append(point[0, 0])
                Y2.append(point[0, 1])
                Z2.append((len(imageSet) - i))
    if cont_art is not None:
        for art in cont_art:
            for point in art:
                X3.append(point[0, 0])
                Y3.append(point[0, 1])
                Z3.append((len(imageSet) - i))

######## VISUALIZATION #########

fig = plt.figure()
ax = plt.axes(projection="3d")

X1 = np.array(X1)  # Lungs
Y1 = np.array(Y1)
Z1 = np.array(Z1)

X2 = np.array(X2)  # Bronchi
Y2 = np.array(Y2)
Z2 = np.array(Z2)

X3 = np.array(X3)  # Artery
Y3 = np.array(Y3)
Z3 = np.array(Z3)

# ax.plot(X2, Y2, Z2, "b,")  # To show bronchi
ax.plot(X1, Y1, Z1, "g,")  # To show lungs
ax.plot(X3, Y3, Z3, "r,")  # To show artery


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.auto_scale_xyz([0, 512], [0, 512], [20, 67])  # To keep proportions
plt.show()

####### END VISUALIZATION #######