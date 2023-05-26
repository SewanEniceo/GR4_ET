import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
  model=tf.keras.models.load_model('wow_card.h5')
  return model
model=load_model()
st.write("""
# Card Classification System by Group 4 """
)
st.text(" Course and Section: CPE 019 - CPE32S3")
st.text(" Members:")
st.text(" Eniceo, Sean Paolo")
st.text(" Fernandez, Rhenz")
st.text(" Sabio, Jedawn")
st.text(" Instructor: Engr. Roman M. Richard")
file=st.file_uploader("Choose card photo from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(224,224)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['ace of clubs', 'ace of diamonds', 'ace of hearts', 
                 'ace of spades', 'eight of clubs', 'eight of diamonds', 
                 'eight of hearts', 'eight of spades', 'five of clubs', 
                 'five of diamonds', 'five of hearts', 'five of spades', 
                 'four of clubs', 'four of diamonds', 'four of hearts', 
                 'four of spades', 'jack of clubs', 'jack of diamonds', 
                 'jack of hearts', 'jack of spades', 'joker', 'king of clubs', 
                 'king of diamonds', 'king of hearts', 'king of spades', 
                 'nine of clubs', 'nine of diamonds', 'nine of hearts', 
                 'nine of spades', 'queen of clubs', 'queen of diamonds', 
                 'queen of hearts', 'queen of spades', 'seven of clubs', 
                 'seven of diamonds', 'seven of hearts', 'seven of spades', 
                 'six of clubs', 'six of diamonds', 'six of hearts', 'six of spades', 
                 'ten of clubs', 'ten of diamonds', 'ten of hearts', 'ten of spades', 
                 'three of clubs', 'three of diamonds', 'three of hearts', 'three of spades', 
                 'two of clubs', 'two of diamonds', 'two of hearts', 'two of spades']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
    
