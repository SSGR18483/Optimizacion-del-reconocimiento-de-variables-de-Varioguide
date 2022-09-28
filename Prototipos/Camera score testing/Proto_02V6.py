#Algoritmo Calificador de imagenes de camaras.
import keras
import matplotlib.pyplot
import numpy as np
import cv2 as cv
import glob
import h5py
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import rcParams
from PIL import Image
from scipy import io

#cargar modelo
model=tf.keras.models.load_model('my_model.h5')
model.summary()

