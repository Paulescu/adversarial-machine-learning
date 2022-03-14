import io

import requests
import streamlit as st
# import pandas as pd
# import numpy as np
from PIL import Image

from src.model import preprocess, inverse_preprocess, load_model, predict
from src.fgsm import fast_gradient_sign, iterative_fast_gradient_sign_

st.title('Adversarial example generator')

doc_markdown = """
## What are adversarial examples? 💡

👉🏽 Do you think it is impossible to fool the vision system of a self-driving Tesla car?

👉🏽 Or that machine learning models used in malware detection software are too good to be evaded by hackers?

👉🏽 Or that face recognition systems in airports are bulletproof?

Like any of us machine learning enthusiasts, you might fall into the trap of thinking that deep models used out there are perfect.

### Well, you are WRONG.

There are easy ways to build **adversarial examples** that can fool any deep learning model and create security issues.

With this app you can create your own adversarial examples, using the **Iterative Fast Gradient Sign Method**, and fool [`Inception-v3`](https://en.wikipedia.org/wiki/Inceptionv3)


"""

# The model
@st.cache
def load_model_():
    return load_model()

model = load_model_()

st.sidebar.title('Iterative FGSM parameters')
st.markdown(doc_markdown)

# Input image
example_url = 'https://github.com/Paulescu/adversarial-machine-learning/blob/main/images/dog.jpg?raw=true'
# url = st.sidebar.text_input('Introduce URL of the initial image 👇🏼', example_url)
st.markdown('## Original image')
url = st.text_input('Introduce URL of the initial image 👇🏼', example_url)

IMAGE_WIDTH = 350

@st.cache
def fetch_image(url):
    response = requests.get(url)
    x = Image.open(io.BytesIO(response.content))
    x = inverse_preprocess(preprocess(x))
    return x
img = fetch_image(url)

prediction = predict(model, img)
caption = f'{prediction["label"]} \n {prediction["confidence"]:.0%}'
st.image(img, caption=caption, width=IMAGE_WIDTH*2)

# Selector parameters
epsilon = st.sidebar.slider('Step size', min_value=0.0, max_value=0.25, step=0.01, value=0.09, format="%.3f")
alpha = st.sidebar.slider('Max perturbation', min_value=0.00, max_value=0.250, step=0.001, value=0.025, format="%.3f")
n_steps = st.sidebar.number_input('Number of steps', step=1, min_value=1, max_value=50, value=9)

# image_width = 299
st.markdown('## FGSM steps')
image_width = int(IMAGE_WIDTH * 2 / 3)
iterator = iterative_fast_gradient_sign_(model, preprocess(img), epsilon, n_steps=n_steps, alpha=alpha)
counter = 1
for x_adv, grad in iterator:

    st.markdown(f'## Step {counter}')

    # get model predictions
    prediction_adv = predict(model, x_adv)

    # print them
    caption_adv = f'= {prediction_adv["label"]} \n {prediction_adv["confidence"]:.0%}'
    st.image([x_adv, grad, x_adv], width=image_width, caption=['', f'* {epsilon}', caption_adv], output_format='JPEG')

    counter += 1

# x_adv, grad = fast_gradient_sign(model, preprocess(img), epsilon, output_type='rgb')
# prediction_adv = predict(model, x_adv)
# for i in range(0, 3):
#     caption_adv = f'= {prediction_adv["label"]} \n {prediction_adv["confidence"]:.0%}'
#     st.image([grad, x_adv], width=image_width, caption=[f'* {epsilon}', caption_adv], output_format='JPEG')

