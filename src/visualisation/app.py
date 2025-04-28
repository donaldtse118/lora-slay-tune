# inbuilt
import logging
import json
import os

# 3rd parties

import torch
import umap
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# local imports
import utils
from model.embed_func import batch_generate_embedding

# for streamlit read dependencies
import sys
sys.path.append("/workspaces/lora-fine_tuning-on-embedding/src")
print(sys.path)

# suppress noise
for quite_path in ["watchdog.observers.inotify_buffer", 'numba.core']:
    quite_logger = logging.getLogger(quite_path)
    quite_logger.setLevel(logging.INFO)

# Set page config to wider layout
st.set_page_config(
    page_title="💬 Gen Z Vibe Check — lora-slay-tune",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎤 Gen Z Slang Visualizer")

# Load models once


@st.cache_resource
def get_models():
    base_model, lora_model, tokenizer = utils.load_models()
    return base_model, lora_model, tokenizer


base_model, ft_model, tokenizer = get_models()

option = st.selectbox(
    'Select a Gen Z phrase:',
    ('Slay', 'No cap', 'Sus')
)

st.write(f'You selected {option}')


def read_data():

    def _read_dataset(dataset_path: str):

        data = []

        with open(dataset_path, "r") as f:
            for line in f:
                row = json.loads(line)
                data.append({'group': row['anchor'],
                            'value': row['anchor'],
                            'type': 'input'
                            })
                data.append({'group': row['anchor'],
                            'value': row['positive'],
                            'type': 'positive'
                            })
                data.append({'group': row['anchor'],
                            'value': row['negative'],
                            'type': 'negative'
                            })

        df = pd.DataFrame(data)
        return df
 
    df_eval = _read_dataset(dataset_path=utils.EVALUATE_DATASET_PATH)
    # read training dataset as well for more data to dimension reduction
    df_train = _read_dataset(dataset_path=utils.TRAINING_DATASET_PATH)
    # df_train = df_train[df_train['group'] == option]

    df = pd.concat([df_eval, df_train], ignore_index=True)

    df.drop_duplicates(subset=['value'], inplace=True)

    return df


def generate_figure(model):

    df = read_data()

    with torch.no_grad():
        df['embs'] = batch_generate_embedding(df['value'], tokenizer, model)

        dimension = ['x', 'y', 'z']

        umap_model = umap.UMAP(n_components=3, random_state=42)
        reduced_embs = umap_model.fit_transform(np.array(df['embs'].tolist()))
        df['reduced_embs'] = reduced_embs.tolist()
        df[dimension] = reduced_embs.tolist()

        # plot selected group only
        df_plot = df[df['group'] == option]
        # skip text too long
        df_plot = df_plot[df_plot['value'].str.len() < 20]


        fig = px.scatter_3d(df_plot, x="x", y="y", z="z",
                            text=df_plot['value'],
                            color=df_plot['type'],
                            title="Gen Z Phrase Embedding Space")

        return fig


if st.button("Visual embedding space"):

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Base Model")
        fig = generate_figure(base_model)
        st.plotly_chart(fig, width=1024, height=768)
    with col2:
        st.subheader("Fine-tuned Model")
        fig = generate_figure(ft_model)
        st.plotly_chart(fig, width=1024, height=768)
