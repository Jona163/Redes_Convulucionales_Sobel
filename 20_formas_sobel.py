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
    
