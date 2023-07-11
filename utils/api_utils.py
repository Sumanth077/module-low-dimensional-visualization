import base64

from io import BytesIO
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import requests
import streamlit as st
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from PIL import Image, ImageDraw
from stqdm import stqdm
from tqdm import tqdm
import umap.umap_ as umap


def is_success(response):
    if response.status.code != status_code_pb2.SUCCESS:
        print(response.status)
        return False
    return True


def is_mixed_success(response):
    if response.status.code != status_code_pb2.MIXED_STATUS:
        print(response.status)
        return False
    return True


def get_one_page_inputs(
    stub: service_pb2_grpc.V2Stub,
    user_app_id_pbf: resources_pb2.UserAppIDSet,
    page: int,
    per_page: int = 1000,
) -> service_pb2.ListInputsRequest:
    list_inputs_response = stub.ListInputs(
        service_pb2.ListInputsRequest(
            user_app_id=user_app_id_pbf, page=page, per_page=per_page
        )
    )
    if not is_success(list_inputs_response) and not is_mixed_success(
        list_inputs_response
    ):
        print(list_inputs_response.status)
        raise Exception(
            "List inputs failed, status: " + list_inputs_response.status.description
        )

    return list_inputs_response


def get_all_inputs_and_concepts(
    stub: service_pb2_grpc.V2Stub, user_app_id_pbf: resources_pb2.UserAppIDSet
) -> Tuple[List[resources_pb2.Input], List[resources_pb2.Input], List[str]]:
    """Function that gets all inputs and their respective concepts from the Clarifai API.

    Args:
        stub (service_pb2_grpc.V2Stub): _description_
        user_app_id_pbf (resources_pb2.UserAppIDSet): _description_

    Returns:
        _type_: _description_
    """
    page = 1
    input_with_concepts_list = []
    input_without_concepts_list = []
    concept_list = []
    while True:
        list_inputs_response = get_one_page_inputs(
            stub, user_app_id_pbf, page, per_page=1000
        )

        # Check if we have no more inputs
        if len(list_inputs_response.inputs) == 0:
            break

        # Find videos that failed processing
        for input in tqdm(list_inputs_response.inputs):
            if len(input.data.concepts) == 1:
                input_with_concepts_list.append(input)
                concept_list.append(input.data.concepts[0].id)
            elif len(input.data.concepts) == 0:
                input_without_concepts_list.append(input)

        # Iterate page to get new data next iteration
        page += 1

    st.info(f"Found {len(input_with_concepts_list)} Inputs with concepts")
    st.info(f"Found {len(input_without_concepts_list)} Inputs without concepts")

    return input_with_concepts_list, input_without_concepts_list, concept_list


def get_input_by_id(stub, user_app_id_pbf: resources_pb2.UserAppIDSet, input_id):
    get_input_response = stub.GetInput(
        service_pb2.GetInputRequest(user_app_id=user_app_id_pbf, input_id=input_id)
    )
    if get_input_response.status.code != status_code_pb2.SUCCESS:
        print(get_input_response.status)
        raise Exception(
            "Get input failed, status: " + get_input_response.status.description
        )
    return get_input_response, input_id


def get_annotations_per_page(stub, user_app_id_pbf: resources_pb2.UserAppIDSet, page):
    list_annotations_request = service_pb2.ListAnnotationsRequest(
        user_app_id=user_app_id_pbf,
        page=page,
        per_page=1000,
        list_all_annotations=True,
        return_model_output=True,
    )

    list_annotations_response = stub.ListAnnotations(list_annotations_request)
    if not is_success(list_annotations_response) and not is_mixed_success(
        list_annotations_response
    ):
        print(list_annotations_response.status)
        raise Exception(
            "List annotations failed, status: "
            + list_annotations_response.status.description
        )
    return list_annotations_response


@st.cache_data
def get_all_embedding_annotations(
    _stub: service_pb2_grpc.V2Stub, userDataObject: resources_pb2.UserAppIDSet
) -> Tuple[List[Dict[str, Union[str, np.ndarray]]], List[Dict[str, str]]]:
    """
    Retrieve all annotations containing embeddings and concepts for a given user data object from the specified stub.

    Args:
        _stub (service_pb2_grpc.V2Stub): The gRPC stub used for communicating with the Clarifai Platform.
        userDataObject (UserAppIDSet): The ID of the user data object.

    Returns:
        Tuple[List[Dict[str, Union[str, np.ndarray]]], List[Dict[str, Union[str]]]]: The embeddings and concepts as a tuple of two lists of dictionaries respectively. The embeddings are represented as numpy ndarrays and the concepts as strings.

    """

    page = 1
    embedding_list = []
    concept_list = []
    while True:
        list_annotations_response = get_annotations_per_page(
            _stub, userDataObject, page
        )

        if len(list_annotations_response.annotations) == 0:
            break

        for annotation in list_annotations_response.annotations:
            if annotation.data.embeddings:
                input_annotations_dict = {}
                input_annotations_dict["input_id"] = annotation.input_id
                input_annotations_dict["embedding"] = np.array(
                    annotation.data.embeddings[0].vector
                )
                embedding_list.append(input_annotations_dict)

            if annotation.data.concepts:
                if annotation.data.concepts[0].name:
                    input_concept_dict = {}
                    input_concept_dict["input_id"] = annotation.input_id
                    input_concept_dict["concept"] = annotation.data.concepts[0].name
                    concept_list.append(input_concept_dict)
        page += 1

    return embedding_list, concept_list


def url_to_embeddable_image(image_url: str):
    """Download the image, convert it to a NumPy array, and then read
    it into OpenCV format

    Args:
        image_url (str): URL path of the image

    Returns:
        np.ndarray: Numpy array containing image information

    """
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        embeddable_image = create_embeddable_image(img)
    except Exception as e:
        print(f"failed to fetch {image_url}")
        print(f"Error: {e}")
        return None

    return embeddable_image


def create_embeddable_image(img: Image) -> str:
    """Function that embed an image to be then displayed on a bokeh plot.

    Args:
        data (np.ndarray): Numpy array containing image information

    Returns:
        str: Base64 string representation of the image
    """
    # data = data.reshape((50, 50, 1))
    try:
        image = img.resize((50, 50), Image.BICUBIC)
        buffer = BytesIO()
        image.save(buffer, format="WEBP")
        for_encoding = buffer.getvalue()
    except Exception as e:
        print(f"Failed to create embeddable image")
        print(f"Error: {e}")
        return None

    return "data:image/png;base64," + base64.b64encode(for_encoding).decode()


# Function for Dimensionality reduction using UMAP
def get_umap_embedding(embedding_list, n_neighbors):
    reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
    reducer.fit(embedding_list)
    embedding = reducer.embedding_
    return embedding


@st.cache_data
def get_base_workflow(_stub, userDataObject: resources_pb2.UserAppIDSet):
    get_workflow_response = _stub.GetWorkflow(
        service_pb2.GetWorkflowRequest(
            user_app_id=userDataObject,
        )
    )

    if get_workflow_response.status.code != status_code_pb2.SUCCESS:
        print(get_workflow_response.status)
        raise Exception(
            "Get workflow failed, status: " + get_workflow_response.status.description
        )

    return get_workflow_response.workflow.id


def get_inputs_per_page(stub, userDataObject, page, per_page=1000):
    list_inputs_response = stub.ListInputs(
        service_pb2.ListInputsRequest(
            user_app_id=userDataObject, page=page, per_page=per_page
        )
    )
    if not is_success(list_inputs_response) and not is_mixed_success(
        list_inputs_response
    ):
        print(list_inputs_response.status)
        raise Exception(
            "List inputs failed, status: " + list_inputs_response.status.description
        )

    return list_inputs_response


def process_list_inputs_response(list_inputs_response):
    input_success_status = {
        status_code_pb2.INPUT_DOWNLOAD_SUCCESS,
        status_code_pb2.INPUT_DOWNLOAD_PENDING,
        status_code_pb2.INPUT_DOWNLOAD_IN_PROGRESS,
    }

    input_dict_list = []
    for idx, input in tqdm(
        enumerate(list_inputs_response.inputs),
        total=len(list_inputs_response.inputs),
        desc="Formatting inputs",
    ):
        if input.status.code not in input_success_status:
            continue

        # Initializations
        input_dict = {}
        input_dict["input_id"] = input.id
        input_dict["url"] = input.data.image.url
        input_dict_list.append(input_dict)

    return input_dict_list


@st.cache_data
def get_all_inputs(_stub, userDataObject):
    page = 1
    input_dict_list = []
    while True:
        list_inputs_response = get_inputs_per_page(_stub, userDataObject, page)

        if len(list_inputs_response.inputs) == 0:
            break

        processed_input_list = process_list_inputs_response(list_inputs_response)
        input_dict_list.extend(processed_input_list)
        page += 1

    return input_dict_list
