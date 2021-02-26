import numpy as np
import cv2

def Media(img, kernel):
    """
    autora: Angelica Rivas
    """
    return cv2.filter2D(src=img, ddepth=-1, kernel=kernel) 

def FiltroGaussiano(img, k, sigma=1):
    """
    autora: Angelica Rivas
    Recibe:
        img: matriz de la imagen
        k: tamanio del kernel
        sigma: sigma para distribución uniforme
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
