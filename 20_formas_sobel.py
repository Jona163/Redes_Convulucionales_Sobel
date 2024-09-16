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
