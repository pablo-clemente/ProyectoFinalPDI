#manejar carpetas
import os

#cargar bibliotecas necesarias
import numpy as np
import cv2

def BinarizarInv(img):
    """
    Autor: Pablo Clemente
    Recibe:
        img: imagen en escala de grises
    Devuelve: 
        imagen binarizada inversa usando opencv
    """
    return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

def Dilate(img, kernel=np.uint8(np.ones((3,3)))):
    """
    Autor: Pablo Clemente
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
    #dilatacion
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
    Autor: Pablo Clemente
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
    Autor: Pablo Clemente
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
    Autor: Pablo Clemente
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
    Autora: Angelica Rivas
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
    Autora: Angelica Rivas
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
    Autor: Pablo Clemente
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
    
def Media(img, kernel):
    """
    Autora: Angelica Rivas
    """
    return cv2.filter2D(src=img, ddepth=-1, kernel=kernel) 

def FiltroGaussiano(img, k, sigma=1):
    """
    Autora: Angelica Rivas
    Recibe:
        img: matriz de la imagen
        k: tamanio del kernel
        sigma: sigma para distribucion uniforme
    Devuelve:
        imagen con filtro de suavisado gaussiano
    """
    ###################
    ##Calcular Kernel##
    ###################
    #calculo la mascara, 0 en el centro, entre mas se aleja mas grande es el numero
    coordenadas = np.arange(-(k // 2), (k // 2)+1)
    kernel = np.zeros((k,k))
    
    #obtener el valor en distribucion normal para cada punto en mascara
    comun = 2*(sigma**2) #es comun en division y exponencial de e
    parte1 = 1/(np.pi * sigma)
    for y in range(k):
        for x in range(k):
            kernel[x,y] = parte1 * (np.e ** (-(coordenadas[x]**2+coordenadas[y]**2)/comun))
    
    return cv2.filter2D(src=img, ddepth=-1, kernel=kernel) 

def SegmentacionDeVasosSanguineos(nombre_imagen_original, nombre_imagen_output, kernel, k, sigma):
    """
    Autores: Pablo Clemente, Angelica Rivas
    Recibe:
        nombre_imagen_original: ruta a la imagen a segmentar
        nombre_imagen_output: ruta donde guardar la imagen segmentada
        kernel: kernel para el filtro de media
        k: tamanio del kernel del filtro gaussiano
        sigma: sigma del filtro gaussiano
    Devuelve:
        nada.
    """
    #lectura de imagen original
    imagen = CargarArchivo(nombre_imagen_original)
    #cargar la capa verde de la imagen
    img_gris = imagen[:,:,1] 
    #obtener imagen suavisada combinando filtrado Media y Gaussiano
    img_blur = Media(img_gris, kernel)
    img_blur = FiltroGaussiano(img_blur,k=3, sigma=.9)
    #ajustar contraste con clahe
    img_ajuste = clahe.apply(img_blur)
    #obtener imagen binarizada inversa
    img_bin_inv = BinarizarInv(img_ajuste)
    #apertura a la imagen binarizada
    img_opening = AperturaImagen(img_bin_inv) 
    #segmentar
    img_seg = SegmentacionWatershed(imagen, np.uint8(img_opening))
    #guardar archivo
    GuardarArchivo(img_seg, nombre_imagen_output)
    return

def CargarArchivo(nombre):
    """
    Autora: Angelica Rivas
    Recibe:
        nombre: ruta de la imagen a cargar
    Devuelve:
        imagen
    """
    imagen = cv2.imread(nombre) #cargar la imagen
    #escalar la imagen a multiplos de 2.
    #imagen = cv2.resize(imagen, (512, 512)) #Solo en caso de pruebas
    return imagen

def GuardarArchivo(img, ruta):
    """
    Autora: Angelica Rivas
    Recibe:
        img: objeto imagen a guardar
        ruta: ruta donde guardar la imagen 
    Devuelve:
        imagen
    """
    cv2.imwrite(ruta, img) 
    
if __name__ == "__main__":
    ###Las rutas donde se obtendran las imagenes y donde se guardaran una vez procesadas
    ruta   = input() #ruta de donde se van a extraer las imagenes
    ruta_procesadas = input() #nuevo folder por cada prueba distinta significativa: "output/"+"prueba/"
    
    ###Obtener los nombres de las imagenes
    nombres = os.listdir(ruta)
    nombres = sorted(nombres)[1:] #evitar el archivo 0 llamado '.ipynb_checkpoints' al empezar en el archivo 1
    
    ###Creacion del ajuste de contraste CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3)) #ventana pequenia para contraste localizado
    
    ###Creacion del kernel
    kernel = np.ones((3,3),np.float32)/13
    
    for nombre in nombres:
        SegmentacionDeVasosSanguineos(nombre_imagen_original=ruta+nombre, nombre_imagen_output=ruta_procesadas+nombre, kernel=kernel, k=3, sigma=.9)
        