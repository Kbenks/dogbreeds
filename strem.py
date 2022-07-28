import streamlit as st
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
Xception_model_aug = keras.models.load_model('./mod')
#ResNet50_model=keras.models.load_model('./dogdetect')
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint             
from tqdm import tqdm
import cv2                
import random
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from glob import glob
from PIL import Image
# load list of dog names
dog_names = ['1.Affenpinscher',
 '2.Afghan_hound',
 '3.Airedale_terrier',
 '4.Akita',
 '5.Alaskan_malamute',
 '6.American_eskimo_dog',
 '7.American_foxhound',
 '8.American_staffordshire_terrier',
 '9.American_water_spaniel',
 '0.Anatolian_shepherd_dog',
 '1.Australian_cattle_dog',
 '2.Australian_shepherd',
 '3.Australian_terrier',
 '4.Basenji',
 '5.Basset_hound',
 '6.Beagle',
 '7.Bearded_collie',
 '8.Beauceron',
 '9.Bedlington_terrier',
 '0.Belgian_malinois',
 '1.Belgian_sheepdog',
 '2.Belgian_tervuren',
 '3.Bernese_mountain_dog',
 '4.Bichon_frise',
 '5.Black_and_tan_coonhound',
 '6.Black_russian_terrier',
 '7.Bloodhound',
 '8.Bluetick_coonhound',
 '9.Border_collie',
 '0.Border_terrier',
 '1.Borzoi',
 '2.Boston_terrier',
 '3.Bouvier_des_flandres',
 '4.Boxer',
 '5.Boykin_spaniel',
 '6.Briard',
 '7.Brittany',
 '8.Brussels_griffon',
 '9.Bull_terrier',
 '0.Bulldog',
 '1.Bullmastiff',
 '2.Cairn_terrier',
 '3.Canaan_dog',
 '4.Cane_corso',
 '5.Cardigan_welsh_corgi',
 '6.Cavalier_king_charles_spaniel',
 '7.Chesapeake_bay_retriever',
 '8.Chihuahua',
 '9.Chinese_crested',
 '0.Chinese_shar-pei',
 '1.Chow_chow',
 '2.Clumber_spaniel',
 '3.Cocker_spaniel',
 '4.Collie',
 '5.Curly-coated_retriever',
 '6.Dachshund',
 '7.Dalmatian',
 '8.Dandie_dinmont_terrier',
 '9.Doberman_pinscher',
 '0.Dogue_de_bordeaux',
 '1.English_cocker_spaniel',
 '2.English_setter',
 '3.English_springer_spaniel',
 '4.English_toy_spaniel',
 '5.Entlebucher_mountain_dog',
 '6.Field_spaniel',
 '7.Finnish_spitz',
 '8.Flat-coated_retriever',
 '9.French_bulldog',
 '0.German_pinscher',
 '1.German_shepherd_dog',
 '2.German_shorthaired_pointer',
 '3.German_wirehaired_pointer',
 '4.Giant_schnauzer',
 '5.Glen_of_imaal_terrier',
 '6.Golden_retriever',
 '7.Gordon_setter',
 '8.Great_dane',
 '9.Great_pyrenees',
 '0.Greater_swiss_mountain_dog',
 '1.Greyhound',
 '2.Havanese',
 '3.Ibizan_hound',
 '4.Icelandic_sheepdog',
 '5.Irish_red_and_white_setter',
 '6.Irish_setter',
 '7.Irish_terrier',
 '8.Irish_water_spaniel',
 '9.Irish_wolfhound',
 '0.Italian_greyhound',
 '1.Japanese_chin',
 '2.Keeshond',
 '3.Kerry_blue_terrier',
 '4.Komondor',
 '5.Kuvasz',
 '6.Labrador_retriever',
 '7.Lakeland_terrier',
 '8.Leonberger',
 '9.Lhasa_apso',
 '0.Lowchen',
 '1.Maltese',
 '2.Manchester_terrier',
 '3.Mastiff',
 '4.Miniature_schnauzer',
 '5.Neapolitan_mastiff',
 '6.Newfoundland',
 '7.Norfolk_terrier',
 '8.Norwegian_buhund',
 '9.Norwegian_elkhound',
 '0.Norwegian_lundehund',
 '1.Norwich_terrier',
 '2.Nova_scotia_duck_tolling_retriever',
 '3.Old_english_sheepdog',
 '4.Otterhound',
 '5.Papillon',
 '6.Parson_russell_terrier',
 '7.Pekingese',
 '8.Pembroke_welsh_corgi',
 '9.Petit_basset_griffon_vendeen',
 '0.Pharaoh_hound',
 '1.Plott',
 '2.Pointer',
 '3.Pomeranian',
 '4.Poodle',
 '5.Portuguese_water_dog',
 '6.Saint_bernard',
 '7.Silky_terrier',
 '8.Smooth_fox_terrier',
 '9.Tibetan_mastiff',
 '0.Welsh_springer_spaniel',
 '1.Wirehaired_pointing_griffon',
 '2.Xoloitzcuintli',
 '3.Yorkshire_terrier',
 '4.Rottweiller',
 '5.Staffordshire_bull_terrier',
 '6.Pomsky',
 '7.Dutch_shepherd',
 '8.Aidi',
 '9.Beldi_moroccan',
 '0.Argentino_dogo',
 '1.Czechoslovakian_Wolfdog',
 '2.Rhodesian_ridgeback',
 '3.Jack_russell_terrier',
 '4.Fila_brasileiro',
 '5.Presa_canario',
 '6.Spanish_mastiff',
 '7.Russiky_toy',
 '8.Boerboel',
 '9.American_pitbull_terrier',
 '0.American_bulldog',
 '1.American_bully',
 '2.Continental_bulldog']

def load_image(image_file):
	img = Image.open(image_file)
	return img
# def extract_VGG19(tensor):
#	from keras.applications.vgg19 import VGG19, preprocess_input
#	return VGG19(weights='imagenet', include_top=False).predict(preprocess_input(tensor))
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("./lfw/*/*"))
random.shuffle(human_files)
                              

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

# load color (BGR) image
image = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def extract_Xception(tensor):
	from keras.applications.xception import Xception, preprocess_input
	return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

### a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
def Xception_predict_breed (img_path):
    # extract the bottle neck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path)) 
    ## get a vector of predicted values
    predicted_vector = Xception_model_aug.predict(bottleneck_feature) 
    
    ## return the breed
    return dog_names[np.argmax(predicted_vector)]

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

#def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
   # img = preprocess_input(path_to_tensor(img_path))
   # return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
#def dog_detector(img_path):
   # prediction = ResNet50_predict_labels(img_path)
   # return ((prediction <= 268) & (prediction >= 151))
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = load_img(image_file, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(image_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(image_paths)]
    return np.vstack(list_of_tensors)

def Xception_predict_breed (img_path):
    # extract the bottle neck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path)) 
    ## get a vector of predicted values
    predicted_vector = Xception_model_aug.predict(bottleneck_feature) 
    
    ## return the breed
    return dog_names[np.argmax(predicted_vector)]

    
def breed_identifier(img_path):
    prediction = Xception_predict_breed(img_path)
    #pred=re.sub(r'\d+','',prediction)
    #if dog_detector(img_path) == True:
       #st.write('picture is a dog')
    return st.write(f"This dog is a {prediction}\n")
    
    if face_detector(img_path) == True:
        st.write('This is a human, "BACHARE" as we Moroccan say')
        return st.write(f"This person looks like a {pred}\n")
        
    
    else:
        return st.write(f'Not sure if it is a dog, if so it is a {pred}\n')

st.set_page_config( page_title="DOG BREED")
st.title("WHAT IS THE BREED OF THIS DOG ?")
 




file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if file is not None:
	breed_identifier(img)
	st.image(load_image(file),width=250)
	

			 

             
    
    
