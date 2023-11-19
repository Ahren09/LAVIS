
import os
import os.path as osp
import platform
import sys

from tqdm import trange

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

sys.path.insert(0, os.path.abspath('/Users/ahren/Workspace/Multimodal/LAVIS/'))
sys.path.insert(0, os.path.abspath('../LLaVA/'))


import torch
import requests
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess

from llava.utils_data import load_data
from arguments import parse_args
from llava.prompts.prompts import get_prompt, get_prompt_for_image_description
from llava.utils import disable_torch_init, print_colored


# ### Load an example image and question

args = parse_args()


img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/projects/pnp-vqa/demo.png' 
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

df, image_directory_name = load_data(args)



PLATFORM = "cpu"

if platform.system() == "Windows":
    device = "cuda:0"
elif platform.system() == "Darwin":
    device = "mps:0"
elif platform.system() == "Linux":
    device = "cuda"
else:
    raise ValueError("Unknown platform.")


model, vis_processors, txt_processors = load_model_and_preprocess(name="pnp_vqa", model_type="base", is_eval=True, device=device)

existing_results = None

def save(answer_li: list):
    with pd.ExcelWriter(args.answers_file) as writer:
        results_df = pd.DataFrame(answer_li)

        if existing_results is not None:
            results_df = pd.concat([existing_results, results_df], ignore_index=True)

        results_df.to_excel(writer, index=False)

# ### Preprocess image and text inputs

START = 0

for idx in trange(START, len(df), args.batch_size):

    line = df.iloc[idx]

    # raw_image = Image.open("./demo.png").convert("RGB")
    question = "What is the black objects on the salad called?"
    print(question)

    question, image_file = get_prompt(args, line)
    raw_image = Image.open(image_file).convert('RGB')

    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    question = txt_processors["eval"](question)

    samples = {"image": image, "text_input": [question]}


    # ### PNP-VQA utilizes 3 submodels to perform VQA:
    # #### 1. Image-Question Matching
    # Compute the relevancy score of image patches with respect to the question using GradCAM
    samples = model.forward_itm(samples=samples)


    # Gradcam visualisation
    dst_w = 720
    w, h = raw_image.size
    scaling_factor = dst_w / w

    resized_img = raw_image.resize((int(w * scaling_factor), int(h * scaling_factor)))
    norm_img = np.float32(resized_img) / 255
    gradcam = samples['gradcams'].reshape(24,24)

    avg_gradcam = getAttMap(norm_img, gradcam, blur=True)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(avg_gradcam)
    ax.set_yticks([])
    ax.set_xticks([])
    print('Question: {}'.format(question))


    # #### 2. Image Captioning
    # Generate question-guided captions based on the relevancy score


    # Ahren: topk on Mac OS X only support 16
    # We need to set both top_k and num_patches to 16

    if torch.cuda.is_available():
        samples = model.forward_cap(samples=samples, num_captions=50)

    else:
        samples = model.forward_cap(samples=samples, num_captions=50, num_patches=16, top_k=16)
    print('Examples of question-guided captions: ')
    samples['captions'][0][:5]


    # #### 3. Question Answering
    # Answer the question by using the captions

    # In[9]:


    pred_answers = model.forward_qa(samples, num_captions=50)
    print('Question: {} \nPredicted answer: {}'.format(question, pred_answers[0]))


    # ### Generate answer by calling `predict_answers()` directly
    #

    # In[10]:


    pred_answers, caption, gradcam = model.predict_answers(samples, num_captions=50, num_patches=20)
    print('Question: {} \nPredicted answer: {}'.format(question, pred_answers[0]))

