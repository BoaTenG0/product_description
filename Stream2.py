# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:29:53 2024

@author: TECH PLUG
"""

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import Model
import io

# Load the vocabulary
vocab = np.load('C:/Users/TECH PLUG/Downloads/Telegram Desktop/vocab1.npy', allow_pickle=True).item()
inverse_vocab = {v: k for k, v in vocab.items()}

# Load the feature extraction model
resnet_model = ResNet50(weights='imagenet')
resnet_model = Model(inputs=resnet_model.inputs, outputs=resnet_model.layers[-2].output)

# Load the trained model
model = load_model('C:/Users/TECH PLUG/Downloads/Telegram Desktop/product_description_generator1.h5')


def preprocess_image(image):
    """Preprocess the image to be fed into ResNet50."""
    # Convert image to RGB if it's not, to ensure it has 3 channels.
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


def extract_features(image, model):
    """Extract features from the image."""
    features = model.predict(image)
    return features

def generate_caption(image, feature_model, caption_model, vocab, inverse_vocab, max_length=14):
    """Generate a caption for the image."""
    image = preprocess_image(image)
    features = extract_features(image, feature_model).reshape(1, 2048)
    caption = ['startofseq']
    for i in range(max_length):
        sequence = [vocab[word] for word in caption if word in vocab]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = caption_model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = inverse_vocab[yhat]
        if word == 'endofseq':
            break
        caption.append(word)
    final_caption = ' '.join(caption[1:])
    return final_caption



# app.py
def main():
    st.sidebar.markdown(
    '<h1 style="color: #FF4F8B; font-size: 26px;">'
    'Caption Generator <span style="font-size: 30px;">👗</span></h1>',unsafe_allow_html=True)    
    st.title("Caption Generator")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)
            
    with col2:
        if uploaded_file is not None:
            if st.sidebar.button('Generate Description'):
                st.write("Generating description...")
                caption = generate_caption(image, resnet_model, model, vocab, inverse_vocab)
                st.write(f"Caption Description: {caption}")
    

if __name__ == "__main__":
    main()
