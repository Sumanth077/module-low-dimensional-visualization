"""
This UI module is meant to display an interactive plot for low dimensional image data.
The data can be with or without concepts. This type of plot is especially useful for classification problems.
"""
# import os
# import pwd

# Set NUMBA_CACHE_DIR to /dev/null on Unix-like systems
# dir = os.path.join(pwd.getpwuid(os.getuid()).pw_dir, ".cache", "numba")
# os.environ["NUMBA_CACHE_DIR"] = dir
# os.makedirs(dir, exist_ok=True)

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import pandas as pd
import streamlit as st
from bokeh.models import CategoricalColorMapper, ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.palettes import Spectral11
from clarifai_utils.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from google.protobuf import json_format
from PIL import Image
from stqdm import stqdm

from utils.api_utils import (
    get_all_embedding_annotations,
    url_to_embeddable_image,
    get_umap_embedding,
    get_base_workflow,
    get_all_inputs,
)

# Import profiler
# import memory_profiler

################################################
# Required in every Clarifai streamlit app
################################################

# Set page layout
st.set_page_config(layout="wide")

# Get authentication objects to be able to interact with Clarifai API
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()

# Display main app name and information
st.image(
    "https://miro.medium.com/max/700/1*lqCz8cut1LfUMTjo8RtYKQ.jpeg",
    width=300,
)
st.title("Low-dimensional visualization of Image Data")
st.write(
    "This is a Streamlit app that displays a low dimensional\
    representation of your high dimensional image data."
)

# Module configurations
percentage_of_data = st.slider("Percentage of data to use", 0, 100, 100)

# Check if app has base workflow General activated
base_workflow = get_base_workflow(stub, userDataObject)
if base_workflow == "General":
    st.success("Your app has the base workflow General activated")
else:
    st.error(
        "Your app needs to have the base workflow General activated and all inputs indexed to use this app"
    )
    st.stop()

n_neighbors = st.number_input(
    "Insert number of neighbors for dimension reduction (2 to 100)",
    min_value=2,
    max_value=100,
    value=4,
    help="This parameter controls how UMAP balances local versus global structure in the data. It does this by constraining the size of the local neighborhood UMAP will\
            look at when attempting to learn the manifold structure of the data. This means that low values of n_neighbors will force UMAP to concentrate on very local structure (potentially to the detriment of the big picture), while large values will push UMAP to look at larger neighborhoods of each point when estimating the manifold structure of the data, losing fine detail structure for the sake of getting the broader of the data.",
)

if st.button("Create Visualization"):
    # Get all embeddings and labels using Clarifai API
    st.info("Fetching all embeddings and labels")
    embedding_dicts, label_dicts = get_all_embedding_annotations(stub, userDataObject)
    input_dicts = get_all_inputs(stub, userDataObject)

    # Create dataframes and concat on input_id
    input_df = pd.DataFrame(input_dicts)
    embed_df = pd.DataFrame(embedding_dicts)
    label_df = pd.DataFrame(label_dicts)

    # merge dataframes on input id
    df_temp = pd.merge(embed_df, input_df, on="input_id", how="left")
    df = pd.merge(df_temp, label_df, on="input_id", how="left")

    # Get and store all input information
    amount_of_data = int(len(df) * percentage_of_data / 100)

    # Take a subset of the data
    df = df.sample(n=amount_of_data, random_state=42).reset_index()

    # Create PIL images from urls
    embeddable_image_list = []
    to_pil_threads = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for input_url in stqdm(
            df["url"].tolist(), total=len(df["url"].tolist()), desc="Rendering images"
        ):
            to_pil_threads.append(
                (executor.submit(url_to_embeddable_image, input_url), input_url)
            )
            time.sleep(0.04)

        for task, input_url in stqdm(
            to_pil_threads, total=len(to_pil_threads), desc="Storing images"
        ):
            image = task.result()
            # if image is not None:
            embeddable_image_list.append(image)

    st.info(f"Creating visualization using {len(embeddable_image_list)} inputs")

    embedding = get_umap_embedding(df["embedding"].tolist(), n_neighbors)

    # Create DataFrame and store all relevant information
    embedding_df = pd.DataFrame(embedding, columns=("x", "y"))

    # Concat embedding_df with df
    embedding_df = pd.merge(
        embedding_df, df, how="inner", left_index=True, right_index=True
    )

    # Rename columns
    embedding_df.rename(
        columns={"input_id": "image_name", "concept": "label"}, inplace=True
    )
    embedding_df["image"] = embeddable_image_list

    # Add Bokeh specific objects
    datasource = ColumnDataSource(embedding_df)

    color_mapping = CategoricalColorMapper(
        factors=[str(x) for x in embedding_df["label"].unique()], palette=Spectral11
    )

    # Create Bokeh figure
    plot_figure = figure(
        title="Embedding Analysis",
        plot_width=800,
        plot_height=800,
        tools=("pan, wheel_zoom, reset"),
    )

    # Add hover feature for bokeh plot. Hover over data point will display image and concept name
    plot_figure.add_tools(
        HoverTool(
            tooltips="""
    <html>
    <head>
        <style>
            div 
            {
            word-wrap:break-word;
            width: 150px;
            border:1px solid black;
            }
        </style>
    </head>
    <div>
        <div>
            <img src='@image' style='float: left; margin: 5px 15px 15px 15px'/>
        </div>
        <div>
            <span style='font-size: 14px'>Concept: @label</span>
        </div>
        <div>
            <span style='font-size: 8px; color: #224499'>Input ID: @image_name</span>
        </div>
    </div>
    """
        )
    )

    # Draw circular datapoint on bokeh plot
    plot_figure.circle(
        "x",
        "y",
        source=datasource,
        color=dict(field="label", transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4,
    )

    # Display bokeh plot on streamlit app
    st.bokeh_chart(plot_figure, use_container_width=True)
