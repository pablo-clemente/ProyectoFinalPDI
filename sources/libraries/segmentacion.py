import numpy as np
import cv2

def BinarizarInv(img):
    """
    autor: Pablo Clemente
    Recibe:
        img: imagen en escala de grises
    Devuelve: 
        imagen binarizada inversa usando opencv
    """
    return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

def Dilate(img, kernel=np.uint8(np.ones((3,3)))):
    """
    autor: Pablo Clemente
    Recibe:
        img: imagen binaria 
        kernel: kernel cuadrado (Uint8). Default 3x3 completa.
    Devuelve: 
        imagen diltada
    """
    S = img.shape
    F = kernel.shape
    R = S[0] + F[0]-1 #row
    C = S[1] + F[1]-1 #column
    N = np.zeros((R,C))
    #padding
    for i in range(S[0]):
        for j in range(S[1]):
            N[i+np.int((F[0]-1)/2),j+np.int((F[1]-1)/2)] = img[i,j]
    #dilatación
    for i in range(S[0]):
        for j in range(S[1]):
            k = N[i:i+F[0],j:j+F[1]]
            result = (k==kernel) #return true or false
            final = np.any(result==True)
            if final:
                img[i,j]=1
            else:
                img[i,j]=0
    return img

def DilatarImagen(img,kernel=np.uint8(np.ones((3,3),np.uint8)),it=1):
    """
    autor: Pablo Clemente
    Recibe:
        img: imagen binaria 
        kernel: kernel cuadrado (Uint8). Default 3x3 completa.
        iter: numero de iteraciones. Default 1.
    Devuelve: 
        imagen diltada
    """
    img = img/255
    for i in range(it):
        imagen = Dilate(img,kernel)
        img= imagen
    return img*255

def Erosion(img, kernel=np.uint8(np.ones((3,3)))):
    """
    autor: Pablo Clemente
    Recibe:
        img: imagen binaria 
        kernel: kernel cuadrado (Uint8). Default 3x3 completa.
    Devuelve: 
        imagen erosionada
    """
    S = img.shape
    F = kernel.shape
    R = S[0] + F[0]-1 #row
    C = S[1] + F[1]-1 #column
    N = np.zeros((R,C))
    #padding
    for i in range(S[0]):
        for j in range(S[1]):
            N[i+np.int((F[0]-1)/2),j+np.int((F[1]-1)/2)] = img[i,j]
    #erosion
    for i in range(S[0]):
        for j in range(S[1]):
            k = N[i:i+F[0],j:j+F[1]]
            result = (k==kernel) #return true or false
            final = np.all(result==True)
            if final:
                img[i,j]=1
            else:
                img[i,j]=0
    return img

def ErosionImagen(img,kernel=np.uint8(np.ones((3,3),np.uint8)),it=1):
    """
    autor: Pablo Clemente
    Recibe:
        img: imagen binaria 
        kernel: kernel cuadrado (Uint8). Default 3x3 completa.
        iter: numero de iteraciones. Default 1.
    Devuelve: 
        imagen erosionada
    """
    img = img/255
    for i in range(it):
        imagen = Erosion(img,kernel)
        img= imagen
    return img*255

def AperturaImagen(img, kernel=np.uint8(np.ones((3,3),np.uint8)), itera=1):
    """
    autora: Angelica Rivas
    Recibe:
        img: imagen binaria 
        kernel: kernel cuadrado (Uint8). Default 3x3 completa.
        iter: numero de iteraciones. Default 1.
    Devuelve: 
        imagen con transformacion de apertura
    """
    img_e = ErosionImagen(img, kernel=kernel, it=itera)
    img_d = DilatarImagen(img_e, kernel=kernel, it=itera)
    return img_d

def CierreImagen(img, kernel=np.uint8(np.ones((3,3))), itera=1):
    """
    autora: Angelica Rivas
    Recibe:
        img: imagen binaria 
        kernel: kernel cuadrado (Uint8). Default 3x3 completa.
        iter: numero de iteraciones. Default 1.
    Devuelve: 
        imagen con transformacion de cierre
    """
    img_d = DilatarImagen(img, kernel=kernel, it=itera)
    img_e = ErosionImagen(img_d, kernel=kernel, it=itera)
    return img_e

def SegmentacionWatershed(imagen,img, kernel=np.uint8(np.ones((3,3)))):
    """
    autor: Pablo Clemente
    Recibe:
        img: imagen original
        img_binary: imagen binaria
        kernel: kernel cuadrado (Uint8). Default 3x3 completa.
    Devuelve: 
        imagen segmentada con algoritmo watershed usando opencv
    """
    img_binary = img.copy()
    # area segura del background
    sure_bg = DilatarImagen(img_binary,kernel,2)
    # encontrar area primer plano
    dist_transform = cv2.distanceTransform(img_binary,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
    # areas desconocidas
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg,dtype = cv2.CV_32F)
    # etiquetar indicadores
    ret, markers = cv2.connectedComponents(sure_fg)
    # Agregar 1 a todas las etiquetas, asi el background no es 0 sino 1
    markers = markers+1
    # Etiquetar la region desconocida con 0 
    markers[unknown==255] = 0
    markers = cv2.watershed(imagen,markers)
    imagen[markers == -1] = [255,0,0]
    return imagen