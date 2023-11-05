
import os
import os.path as osp
import sys

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

sys.path.insert(0, os.path.abspath('/Users/ahren/Workspace/Multimodal/LAVIS/'))
sys.path.insert(0, os.path.abspath('../LLaVA/'))


import torch
import requests
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess

from llava.utils_data import load_data
from arguments import parse_args

# ### Load an example image and question

args = parse_args()


img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/projects/pnp-vqa/demo.png' 
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

data = load_data(args)

raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
# raw_image = Image.open("./demo.png").convert("RGB")
question = "What is the black objects on the salad called?"
print(question)


# In[11]:


# setup device to use
device = "mps:0"


# ### Load PNP-VQA model

# In[12]:


model, vis_processors, txt_processors = load_model_and_preprocess(name="pnp_vqa", model_type="base", is_eval=True, device=device)


# ### Preprocess image and text inputs

# In[13]:


image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
question = txt_processors["eval"](question)

samples = {"image": image, "text_input": [question]}


# ### PNP-VQA utilizes 3 submodels to perform VQA:
# #### 1. Image-Question Matching 
# Compute the relevancy score of image patches with respect to the question using GradCAM

# In[14]:


samples = model.forward_itm(samples=samples)


# In[17]:


dir(raw_image)


# In[15]:


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

# In[8]:

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

