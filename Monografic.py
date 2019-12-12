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


#### FILTERING FUNCTIONS ####
def filterContoursLungs(contours, hierarchy): # Recibe contornos, devuelve los que considera que pertenecen a los pulmones
    toDelete = []

    for i in range(0,len(hierarchy[0]),1): # Escogemos solo los bordes exteriores (sin contar el más exterior)
        if hierarchy[0,i,3] == -1:
            toDelete.append(i)

   # cont = np.delete(contours, toDelete, axis=0)
   # toDelete = []

    i = 0

    for cnt in contours:
        if len(cnt) < 120: # Eliminamos aquellos que consideremos que son demasiado pequeños para pertenecer a un pulmón
            toDelete.append(i)
        i = i + 1

    cont = np.delete(contours, toDelete, axis=0)

    return cont

def filterContoursArtery(contours, hierarchy, last_img): # Recibe contornos, devuelve los que considera que pertenecen a la arteria
    toReturn = []
    toDelete = []
    for i in range(0, len(hierarchy[0]), 1):  # Escogemos solo los bordes exteriores (sin contar el más exterior)
        if hierarchy[0, i, 3] == -1:
            toDelete.append(i)
    cont = np.delete(contours, toDelete, axis=0)

    for cnt in cont: # De entre los contornos interiores
        centro, radio = cv.minEnclosingCircle(cnt)
        redondez = cv.contourArea(cnt) / (np.pi * radio ** 2)
      #  circularity = (4 * math.pi * cv.contourArea(cnt)) / (
      #              cv.arcLength(cnt, 1) * cv.arcLength(cnt, 1))  # Cambiar por nueva fórmula !!!
        if (((cv.contourArea(cnt) < 1000) and (cv.contourArea(cnt) > 100)) and (redondez > 0.8)) or (solape(cnt,last_img)): # REDUCIR LA FACILIDAD CON LA QUE SALEN MOTAS ROJAS DENTRO PULMONES
            toReturn.append(cnt)

    return toReturn

def solape(cnt, last_img):
    nueva = np.zeros(last_img.shape)
    cv.drawContours(nueva, [cnt], 0, (255, 255, 255), -1)
    nueva2 = nueva * last_img

    return cv.norm(nueva2, cv.NORM_L1) > 0

def filterContoursBronq(contours, hierarchy): # Filtramos los contornos, dejando los que creamos que pertenecen a los bronquios
    toReturn = []

   # for i in range(0,len(hierarchy[0]),1): # No queremos el borde exterior (pulmón)
   #     if hierarchy[0,i,3] == -1:
   #         toDelete.append(i)

    for cnt in contours:
        if cv.contourArea(cnt) > 10: # Eliminamos aquellos que consideremos que son demasiado pequeños para pertenecer a un pulmón
            toReturn.append(cnt)

    return toReturn

#### DETECTING FUNCTION ####

def show_img(img, last_img, mode=0, erode=2, dilate=2):

    conts_lung = np.empty([1,2]) # Inicializamos las matrices que contendrán los pulmones y/o los bronquios
    conts_bronq = None
    conts_art = None
    next_img = np.zeros(img.shape)

    if mode==0: # Modo de umbralizado + contornos
        #cv.imshow("pre_thresh",img)
        ret, th2 = cv.threshold(img, 30, 255, cv.THRESH_BINARY) # Umbralizado
        contours, hierarchy = cv.findContours(th2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # Seleccion de contornos

        conts_lung = filterContoursLungs(contours, hierarchy) # Filtramos contornos pulmones

        cv_img = cv.cvtColor(img,cv.COLOR_GRAY2BGR) # th2 -> ver la umbralizada, img -> ver la original

        cv.drawContours(cv_img, conts_lung, -1, (0, 255, 0), 2) # Dibujamos los contornos en la imagen original
       # cv.imshow("post_thresh", th2)
        cv.imshow("window",cv_img)

    elif mode==1: # Modo de umbralizado + contornos + erosión y dilatación
        ret, th2 = cv.threshold(img, 30, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8) # Kernel para dilatación/erosión
        th2 = cv.erode(th2, kernel, iterations=erode) # Erosión
        th2 = cv.dilate(th2, kernel, iterations=dilate) # Dilatación
        contours, hierarchy = cv.findContours(th2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.imshow("original",img)
        conts_lung = filterContoursLungs(contours, hierarchy)

        cv_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # th2 -> ver la umbralizada, img -> ver la original

        cv.drawContours(cv_img, conts_lung, -1, (0, 255, 0), 2)
        cv.imshow("window", cv_img)

    elif mode==2: # Modo de umbralizado + contornos + erosión y dilatación + detección de arteria aorta

        # Step 1. Arteria

        ret, th_art = cv.threshold(img, 5, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        th_art = cv.morphologyEx(th_art, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # ELIMINAMOS RUIDO
        img_arteria = th_art[190:400, 175:325] # Cortamos una sección de la imagen, aquella donde aparece siempre la arteria aorta
        img_arteria = cv.erode(img_arteria, kernel)
        contours_art, hierarchy_art = cv.findContours(img_arteria, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

     #   cv.cvtColor(img_arteria,cv.COLOR_GRAY2BGR)
    #    cv.drawContours(img_arteria, contours_art, -1, (128,128,0), 2)
    #    cv.imshow("arteria",img_arteria)
    #    cv.waitKey(0)

        conts_art = [cnt + (175, 190) for cnt in contours_art]  # Para compensar el slice anterior
        conts_art = filterContoursArtery(conts_art, hierarchy_art, last_img) # Filtramos los contornos para quedarnos siempre con la arteria aorta

        # Aquí ya hemos obtenido el contorno de la arteria

        # Pasos para arteria: Umbralizar, eliminar ruido, extraer contornos, filtrar contornos


        # Step 2. Pulmones

        ret, th_lung = cv.threshold(img, 30, 255, cv.THRESH_BINARY)

        cv.imshow("preborrado_arteria",th_lung)

        cv.fillPoly(th_lung, pts = conts_art, color=(255,255,255)) # Coloreamos de blanco lo que antes era la arteria

        cv.imshow("postborrado_arteria", th_lung)

        th_lung = cv.erode(th_lung, kernel, iterations=erode) # Erosion y dilatacion imagen pulmon
        th_lung = cv.dilate(th_lung, kernel, iterations=dilate)
        contours_lung, hierarchy_lung = cv.findContours(th_lung, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        conts_lung = filterContoursLungs(contours_lung, hierarchy_lung)

        # Step 3. Dibujar los contornos

        cv_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # th_x -> ver la umbralizada, img -> ver la original

        cv.drawContours(cv_img, conts_art, -1, (0, 0, 255), 2) # Contornos de arteria (rojo)
        cv.drawContours(next_img,conts_art,-1,(255,255,255),-1) # Añadimos la arteria actual a esta imagen, para en el siguiente paso tenerla en cuenta
        cv.drawContours(cv_img, conts_lung, -1, (0, 255, 0), 2) # Contornos de pulmones (azul)
        cv.imshow("window", cv_img)

    elif mode==3: # Modo de umbralizado + contornos + erosión y dilatación + detección de arteria aorta + detección de bronquios

        # Step 1. Arteria

        ret, th_art = cv.threshold(img, 5, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        th_art = cv.morphologyEx(th_art, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))  # ELIMINAMOS RUIDO
        img_arteria = th_art[190:400,
                      175:325]  # Cortamos una sección de la imagen, aquella donde aparece siempre la arteria aorta
        contours_art, hierarchy_art = cv.findContours(img_arteria, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        #    cv.cvtColor(img_arteria,cv.COLOR_GRAY2BGR)
        #    cv.drawContours(img_arteria, contours_art, -1, (128,128,0), 2)
        #    cv.imshow("arteria",img_arteria)
        #   cv.waitKey(0)

        conts_art = [cnt + (175, 190) for cnt in contours_art]  # Para compensar el slice anterior
        conts_art = filterContoursArtery(conts_art, hierarchy_art,
                                         last_img)  # Filtramos los contornos para quedarnos siempre con la arteria aorta
        # Aquí ya hemos obtenido el contorno de la arteria

        # Step 2. Pulmones

        ret, th_lung = cv.threshold(img, 30, 255, cv.THRESH_BINARY)
        cv.fillPoly(th_lung, pts=conts_art, color=(255, 255, 255))  # Coloreamos de blanco lo que antes era la arteria
        th_lung = cv.erode(th_lung, kernel, iterations=erode)  # Erosion y dilatacion imagen pulmon
        th_lung = cv.dilate(th_lung, kernel, iterations=dilate)
        contours_lung, hierarchy_lung = cv.findContours(th_lung, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        conts_lung = filterContoursLungs(contours_lung, hierarchy_lung)

        # Step 3. Bronquios

        # 3.1. Primero dibujamos lo que ya tenemos, porque queremos dibujar los bronquios poco a poco y que queden por encima
        cv_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # th2 -> ver la umbralizada, img -> ver la original

        cv.drawContours(cv_img, conts_art, -1, (0, 0, 255), 2)
        cv.drawContours(next_img, conts_art, -1, (255, 255, 255),-1)  # Añadimos la arteria actual a esta imagen, para en el siguiente paso tenerla en cuenta
        cv.drawContours(cv_img, conts_lung, -1, (0, 255, 0), 2)


        count = 1 # Para dibujar la ventana de este pulmon

        # 3.2. Encontrar los bronquios y dibujarlos
        if (conts_lung.size > 0):
            for c in conts_lung: # Por cada pulmón
                maxIzq = tuple(c[c[:, :, 0].argmin()][0])[0] # Obtenemos las coordenadas más extremas de ese pulmón
                maxDer = tuple(c[c[:, :, 0].argmax()][0])[0] # Solo cogeremos contornos de bronquios que estén dentro
                maxTop = tuple(c[c[:, :, 1].argmin()][0])[1]
                maxBot = tuple(c[c[:, :, 1].argmax()][0])[1]

                img_bronq = img[maxTop:maxBot, maxIzq:maxDer] # Seleccionamos solo la imagen de dentro del pulmón

                ret, th_bronq = cv.threshold(img_bronq, 12, 255, cv.THRESH_BINARY_INV)

               # cv.imshow("wind" + str(count), th_bronq)

                contours_bronq, hierarchy_bronq = cv.findContours(th_bronq, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                conts_bronq = filterContoursBronq(contours_bronq, hierarchy_bronq)
                conts_bronq = [cnt + (maxIzq, maxTop) for cnt in conts_bronq]  # Para compensar el slice anterior
                cv.drawContours(cv_img, conts_bronq, -1, (255, 0, 0), 1) # Dibujamos el contorno de los bronquios encontrados

                count = count + 1 # Para dibujar la ventana de este pulmon

        cv.imshow("window", cv_img)

    elif mode==4: # Modo de umbralizado + contornos + erosión y dilatación + detección de arteria aorta + detección de bronquios con umbralizado adaptativo

        # Step 1. Arteria

        ret, th_art = cv.threshold(img, 5, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        th_art = cv.morphologyEx(th_art, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))  # ELIMINAMOS RUIDO
        img_arteria = th_art[190:400,
                      175:325]  # Cortamos una sección de la imagen, aquella donde aparece siempre la arteria aorta
        contours_art, hierarchy_art = cv.findContours(img_arteria, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        #    cv.cvtColor(img_arteria,cv.COLOR_GRAY2BGR)
        #    cv.drawContours(img_arteria, contours_art, -1, (128,128,0), 2)
        #    cv.imshow("arteria",img_arteria)
        #   cv.waitKey(0)

        conts_art = [cnt + (175, 190) for cnt in contours_art]  # Para compensar el slice anterior
        conts_art = filterContoursArtery(conts_art, hierarchy_art,
                                         last_img)  # Filtramos los contornos para quedarnos siempre con la arteria aorta
        # Aquí ya hemos obtenido el contorno de la arteria

        # Step 2. Pulmones

        ret, th_lung = cv.threshold(img, 30, 255, cv.THRESH_BINARY)
        cv.fillPoly(th_lung, pts = conts_art, color=(255,255,255)) # Coloreamos de blanco lo que antes era la arteria
        th_lung = cv.erode(th_lung, kernel, iterations=erode) # Erosion y dilatacion imagen pulmon
        th_lung = cv.dilate(th_lung, kernel, iterations=dilate)
        contours_lung, hierarchy_lung = cv.findContours(th_lung, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        conts_lung = filterContoursLungs(contours_lung, hierarchy_lung)

        # Step 3. Bronquios

        # 3.1. Primero dibujamos lo que ya tenemos, porque queremos dibujar los bronquios poco a poco y que queden por encima
        cv_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # th2 -> ver la umbralizada, img -> ver la original

        cv.drawContours(cv_img, conts_art, -1, (0, 0, 255), 2)
        cv.drawContours(next_img, conts_art, -1, (255, 255, 255), -1)  # Añadimos la arteria actual a esta imagen, para en el siguiente paso tenerla en cuenta
        cv.drawContours(cv_img, conts_lung, -1, (0, 255, 0), 2)

        # 3.2. Obtener la máscara de los pulmones y su norma
        mascara = np.zeros((512, 512), np.uint8)

        cv.fillPoly(mascara, pts=conts_lung, color=(255, 255, 255))  # Coloreamos de blanco lo que corresponde al pulmón, formando la mascara
        cv.imshow("mascara", mascara)

        norma_mascara = cv.norm(mascara, cv.NORM_L1)
        # 3.3. Obtener la norma de la imagen cubierta por la máscara
        norma_img = cv.norm(img, cv.NORM_L1, mascara)

        # 3.4. Obtener el valor medio de gris del interior de los pulmones
      #  print("norma_img",norma_img, "norma_mascara",norma_mascara)

        if (norma_mascara > 0.0):
            grey_lungs = float(norma_img) / (float(norma_mascara) / 255) # En grey_lungs tenemos el valor medio de gris



            # 3.2. Encontrar los bronquios y dibujarlos
            if (conts_lung.size > 0):
                for c in conts_lung: # Por cada pulmón
                    maxIzq = tuple(c[c[:, :, 0].argmin()][0])[0] # Obtenemos las coordenadas más extremas de ese pulmón
                    maxDer = tuple(c[c[:, :, 0].argmax()][0])[0] # Solo cogeremos contornos de bronquios que estén dentro
                    maxTop = tuple(c[c[:, :, 1].argmin()][0])[1]
                    maxBot = tuple(c[c[:, :, 1].argmax()][0])[1]

                    img_bronq_0 = img # Copio la imagen original
                  #  cv.imshow("img_bronq_0",img_bronq_0)
                  #  cv.imshow("mascara",(255 - mascara))
                    img_bronq_0 = cv.add(img_bronq_0, (255 - mascara)) # Elimino de esa imagen lo que no corresponde a los pulmones
                                                                        # Dado que tenemos la mascara guardada, podemos hacerlo

                 #   cv.imshow("result", img_bronq_0)
                    img_bronq = img_bronq_0[maxTop:maxBot, maxIzq:maxDer] # Seleccionamos solo la imagen de dentro del pulmón


                    ret, th_bronq = cv.threshold(img_bronq, grey_lungs * 1.5, 255, cv.THRESH_BINARY_INV) # Antes era 12
                   # cv.imshow("bronq", th_bronq)
                    contours_bronq, hierarchy_bronq = cv.findContours(th_bronq, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    conts_bronq = filterContoursBronq(contours_bronq, hierarchy_bronq)
                    conts_bronq = [cnt + (maxIzq, maxTop) for cnt in conts_bronq]  # Para compensar el slice anterior
                    cv.drawContours(cv_img, conts_bronq, -1, (255, 0, 0), 1) # Dibujamos el contorno de los bronquios encontrados

        cv.imshow("window", cv_img)
    cv.waitKey(0)
    return conts_lung, conts_art, conts_bronq, next_img

################## VARIABLES INICIALES #######################

PATH = [] # Lista de los paths a los tres sets de imágenes
PATH.append("/home/jaime/Documents/SIB Imagenes seleccionadas/Case 1/") # COMPROBAR SI SE PUEDE HACER RUTA RELATIVA AL .PY !!!
PATH.append("/home/jaime/Documents/SIB Imagenes seleccionadas/Case 2/")
PATH.append("/home/jaime/Documents/SIB Imagenes seleccionadas/Case 3/")

imageSet = [] # Lista que contendrá los sets de imágenes
imageSetNumber = int(easygui.buttonbox("Selecciona un set de imágenes:","Select image set",("0","1","2"))) # Índice que determina con qué set trabajaremos
mode = int(easygui.buttonbox("Selecciona un modo:\n0: Umbralizado + contornos\n1: 0 + Erosión y dilatación\n2: 1 + Detección arteria\n3: 2 + Detección bronquios\n4: 3 + Umbralizado adaptativo de bronquios","Select mode set",("0","1","2","3","4"))) # Índice que determina con qué modo trabajaremos


image_files = glob.glob(PATH[imageSetNumber]+"*.dcm")
image_files.sort()
imageSet = [(dicom.dcmread(img)).pixel_array for img in image_files]

################### NORMALIZACIÓN ######################

maxGrey = 0
minGrey = 9999999 # Basta con ser mayor de 65536

for i in range(0,len(imageSet)): # Obtain the maximum and minimum value of grey
    greyValue = np.amax(imageSet[i])
    if (greyValue > maxGrey):
        maxGrey = greyValue

    greyValue = np.amin(imageSet[i])

    if (greyValue < minGrey):
        minGrey = greyValue


factor= 255.0/(maxGrey - minGrey)
for i in range(0,len(imageSet)): # Normalización
   imageSet[i] = ((imageSet[i] - minGrey) * factor).astype(np.uint8)

  # print(np.amax(imageSets[imageSet][i]- minGrey),np.amin(imageSets[imageSet][i]))



################# OBTENCION CONTORNOS #####################

# Set 1 (Pulmón der)
X1 = []
Y1 = []
Z1 = []
# Set 2 (Bronquios)
X2 = []
Y2 = []
Z2 = []
# Set 3 (Arteria)
X3 = []
Y3 = []
Z3 = []



last_img = np.zeros(imageSet[0].shape)

for i in range(0,len(imageSet)): # Obtain contours and visualize images
    print(i)
    conts_lung, conts_art, conts_bronq, next_img = show_img(imageSet[i],last_img,mode) # Mostramos la imagen con los contornos señalados y obtenemos el contorno (Separar en dos funciones?)
    last_img = next_img
    for lung in conts_lung: # Por cada pulmón
        for point in lung: # Por cada punto en el pulmón
                X1.append(point[0,0])
                Y1.append(point[0,1])
                Z1.append((len(imageSet) - i))
    if (conts_bronq is not None):
        for bronq in conts_bronq:
            for point in bronq:
                    X2.append(point[0,0])
                    Y2.append(point[0,1])
                    Z2.append((len(imageSet) - i))
    if (conts_art is not None):
        for art in conts_art:
            for point in art:
                    X3.append(point[0,0])
                    Y3.append(point[0,1])
                    Z3.append((len(imageSet) - i))

######## VISUALIZATION #########


fig = plt.figure()
ax = plt.axes(projection="3d")

X1 = np.array(X1) # Pulmones
Y1 = np.array(Y1)
Z1 = np.array(Z1)

X2 = np.array(X2) # Bronquios
Y2 = np.array(Y2)
Z2 = np.array(Z2)

X3 = np.array(X3) # Arteria
Y3 = np.array(Y3)
Z3 = np.array(Z3)

#ax.voxels(voxels) # PARA REPRESENTAR VOXELS
ax.plot(X2, Y2, Z2, "b,") # PARA REPRESENTAR BRONQUIOS
ax.plot(X1, Y1, Z1, "g,") # PARA REPRESENTAR PULMONES
ax.plot(X3, Y3, Z3, "r,") # PARA REPRESENTAR PULMONES


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.auto_scale_xyz([0, 512], [0, 512], [20, 67]) # Para mantener las proporciones
plt.show()

####### END VISUALIZATION #######