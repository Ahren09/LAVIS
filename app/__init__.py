"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import os.path as osp

import platform

from PIL import Image
import requests

import streamlit as st
import torch


@st.cache()
def load_demo_image():
    img_url = (
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    )
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if platform.system() == 'Windows':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif platform.system() == 'Linux':
    if osp.exists('/nethome/yjin328'):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


    else:
        cache_root = "/media/ahren/Dataset/data/CV/coco2014"

elif platform.system() == "Darwin":
    cache_root = ""


os.makedirs(cache_root, exist_ok=True)

