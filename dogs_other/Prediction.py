from tensorflow import keras
model = keras.models.load_model('./saved_models')
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.utils import img_to_array
from keras.utils import load_img
from keras.callbacks import ModelCheckpoint             
from tqdm import tqdm
import cv2                
import matplotlib.pyplot as plt  

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("./dogImages/train/*/"))]

