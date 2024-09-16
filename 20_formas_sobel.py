# Autor: Jonathan Hernández
# Fecha: 16 Septiembre 2024
# Descripción: Código para procesamiento de imagenes con Sobel 20 formas.
# GitHub: https://github.com/Jona163

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def aplicar_variaciones_sobel(imagen):
    variaciones = []
    titulos = []

    # 1. Variaciones en el tamaño del kernel
    for ksize in [3, 5, 7]:
        sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=ksize)
        variaciones.append(sobel_x)
        titulos.append(f'Sobel X - Kernel {ksize}')

        sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=ksize)
        variaciones.append(sobel_y)
        titulos.append(f'Sobel Y - Kernel {ksize}')

     # 2. Variaciones en el orden de la derivada
    sobel_x_2nd_order = cv2.Sobel(imagen, cv2.CV_64F, 2, 0, ksize=3)
    variaciones.append(sobel_x_2nd_order)
    titulos.append('Sobel X - 2nd Order')

    sobel_y_2nd_order = cv2.Sobel(imagen, cv2.CV_64F, 0, 2, ksize=3)
    variaciones.append(sobel_y_2nd_order)
    titulos.append('Sobel Y - 2nd Order')

    # 3. Diferentes combinaciones de Sobel X e Y
    sobel_combined_1 = cv2.addWeighted(sobel_x, 1, sobel_y, 1, 0)
    variaciones.append(sobel_combined_1)
    titulos.append('Sobel X + Sobel Y (1:1)')

    sobel_combined_2 = cv2.addWeighted(sobel_x, 0.5, sobel_y, 1.5, 0)
    variaciones.append(sobel_combined_2)
    titulos.append('Sobel X + Sobel Y (0.5:1.5)')

    # 4. Métodos de borde diferentes
    sobel_border_default = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_DEFAULT)
    variaciones.append(sobel_border_default)
    titulos.append('Sobel X - Border Default')

    sobel_border_reflect = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    variaciones.append(sobel_border_reflect)
    titulos.append('Sobel X - Border Reflect')

    # 5. Diferentes profundidades de imagen
    sobel_x_16s = cv2.Sobel(imagen, cv2.CV_16S, 1, 0, ksize=3)
    variaciones.append(sobel_x_16s)
    titulos.append('Sobel X - CV_16S')

    sobel_x_8u = cv2.Sobel(imagen, cv2.CV_8U, 1, 0, ksize=3)
    variaciones.append(sobel_x_8u)
    titulos.append('Sobel X - CV_8U')

    # 6. Aplicar Sobel en cada canal de color (si la imagen tiene más de 1 canal)
    if len(imagen.shape) == 3:  # Imagen a color
        sobel_r = cv2.Sobel(imagen[:,:,0], cv2.CV_64F, 1, 0, ksize=3)
        sobel_g = cv2.Sobel(imagen[:,:,1], cv2.CV_64F, 1, 0, ksize=3)
        sobel_b = cv2.Sobel(imagen[:,:,2], cv2.CV_64F, 1, 0, ksize=3)

        variaciones.append(sobel_r)
        titulos.append('Sobel R - Canal Rojo')

        variaciones.append(sobel_g)
        titulos.append('Sobel G - Canal Verde')

        variaciones.append(sobel_b)
        titulos.append('Sobel B - Canal Azul')

    # 7. Aplicar Sobel tras suavizado con filtro Gaussiano
    imagen_suavizada = cv2.GaussianBlur(imagen, (3, 3), 0)
    sobel_suavizado = cv2.Sobel(imagen_suavizada, cv2.CV_64F, 1, 0, ksize=3)
    variaciones.append(sobel_suavizado)
    titulos.append('Sobel X - Imagen Suavizada')

    # 8. Sobel aplicado después de un filtro Laplaciano
    laplaciano = cv2.Laplacian(imagen, cv2.CV_64F)
    sobel_laplaciano = cv2.Sobel(laplaciano, cv2.CV_64F, 1, 0, ksize=3)
    variaciones.append(sobel_laplaciano)
    titulos.append('Sobel X - Laplaciano')
