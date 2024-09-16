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
