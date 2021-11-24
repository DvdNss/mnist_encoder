# coding:utf-8
"""
Filename: app.py
Author: @DvdNss

Created on 11/23/2021
"""

import streamlit as st
from torchvision.transforms import ToTensor

from model import Model


@st.cache(allow_output_mutation=True)
def load_model_and_data():
    train_data, test_data, train_dataloader, test_dataloader = Model.load_mnist(transform=ToTensor(), batch_size=1)
    model = Model(load_model='model/model.pt', img_chan_size=100, global_chan_size=50)
    return test_data, model


# Init page config
page_config = st.set_page_config(
    page_title='Multi-Channel Auto-Encoder for MNIST')
title = st.title('Multi-Channel Auto-Encoder for MNIST')

# Load data and model with cache
test_data, model = load_model_and_data()

c1, c2, c3 = st.columns([1, 2, 1])
c2.image('resources/diag.png', caption='[GitHub](https://github.com/DvdNss/mnist_encoder)', width=300,
         use_column_width=True)

# Choose example number
example = st.number_input('', min_value=0, max_value=len(test_data), help='Choose an example in MNIST test data. ')

# Inference on given example with model
target, prediction, recursive_pred = model.infer(eval_data=test_data, path='example/', n_example=example + 1,
                                                 random=False)

# Split into columns
col1, col2, col3 = st.columns(3)
st.write(f"TARGET: {int(target)}")
st.write(f"PREDICTION: {int(prediction)}")
st.write(f"RECURSIVE: {int(recursive_pred)}")

col1.image('example/target.png', caption='input', width=125)
col2.image('example/output.png', caption='output', width=125)
col3.image('example/output_rec.png', caption='recursive output', width=125)
