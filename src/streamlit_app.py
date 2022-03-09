import io

import requests
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from src.model import preprocess, inverse_preprocess, load_model, predict
from src.fgsm import fast_gradient_sign, iterative_fast_gradient_sign_

st.title('Adversarial example generator')


# The model
@st.cache
def load_model_():
    return load_model()

model = load_model_()

# Input image
example_url = 'https://github.com/Paulescu/adversarial-machine-learning/blob/main/images/dog.jpg?raw=true'
url = st.sidebar.text_input('URL of the image', example_url)

@st.cache
def fetch_image(url):
    response = requests.get(url)
    x = Image.open(io.BytesIO(response.content))
    x = inverse_preprocess(preprocess(x))
    return x
img = fetch_image(url)

prediction = predict(model, img)
caption = f'{prediction["label"]} \n {prediction["confidence"]:.0%}'
st.image(img, caption=caption, width=299*2)

# Selector parameters
epsilon = st.sidebar.slider('Epsilon', min_value=0.0, max_value=0.25, step=0.01, value=0.09, format="%.3f")
alpha = st.sidebar.slider('Alpha', min_value=0.00, max_value=0.250, step=0.001, value=0.025, format="%.3f")
n_steps = st.sidebar.number_input('Number of steps', step=1, min_value=1, max_value=50, value=9)
st.sidebar.image('https://camo.githubusercontent.com/49c80c79c674e543c2c7c2ee7930cc15791f4bd56da17c4b3c91c273349bef8d/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6d656469756d2d2532333132313030452e7376673f267374796c653d666f722d7468652d6261646765266c6f676f3d6d656469756d266c6f676f436f6c6f723d7768697465')

image_width = 299

iterator = iterative_fast_gradient_sign_(model, preprocess(img), epsilon, n_steps=n_steps, alpha=alpha)
for x_adv, grad in iterator:

    # get model predictions
    prediction_adv = predict(model, x_adv)

    # print them
    caption_adv = f'= {prediction_adv["label"]} \n {prediction_adv["confidence"]:.0%}'
    st.image([grad, x_adv], width=image_width, caption=[f'* {epsilon}', caption_adv], output_format='JPEG')

# x_adv, grad = fast_gradient_sign(model, preprocess(img), epsilon, output_type='rgb')
# prediction_adv = predict(model, x_adv)
# for i in range(0, 3):
#     caption_adv = f'= {prediction_adv["label"]} \n {prediction_adv["confidence"]:.0%}'
#     st.image([grad, x_adv], width=image_width, caption=[f'* {epsilon}', caption_adv], output_format='JPEG')

