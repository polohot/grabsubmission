# Grab Submission (Test accuracy 92.2%)
This model is trained on stanford car dataset<br>
https://ai.stanford.edu/~jkrause/cars/car_dataset.html<br>
# Main notebook<br>
<a href="https://github.com/polohot/grabsubmission/blob/master/grab_challange/GRAB%20Image%20classification%20challange.ipynb">grab_challange/GRAB Image classification challange.ipynb</a><br>
# Environment Setup (Tested on windows)
**Please follow the step one by one**<br><br>
**a) Create new environment python = 3.6**
```
conda create -n grab_image python=3.6
```
**b) Activate environment**
```
conda activate grab_image
```
**c) Download and install pytorch**<br>
1. go to pytorch website https://pytorch.org/<br>
2. choose stable and your cuda version<br>
3. copy the link and paste into anaconda terminal<br>

**d) Install FastAI Library**
```
conda install -c pytorch -c fastai fastai
```
**e) Install Jupyter**
```
conda install -c anaconda jupyter 
```
# Pretrained Model Setup<br>
please download the pretrained model in this link <br>
https://drive.google.com/open?id=1bXHQwkWT5fLlCq0Uye8G-kF7RoKTlI4q <br>
paste the file in this directory after clone <br> 
```
grabchallange/data/
```
# Make prediction of test set<br>
(go to section 3.2 of the notebook)<br>

1. Download test data (car_ims) from <br>
http://imagenet.stanford.edu/internal/car196/cars_annos.mat<br>
2. Unzip to **grabsubmission/data/car_ims/***
3. Import library at the top of the notebook
4. Go to section 3.2 of notebook, follow line by line
# Make prediction of hold_out set <br>
(go to section 3.3 of the notebook)<br>
Last section of the notebook states on how to make prediction on the new images using the model above<br><br>
1.Copy pictures into hold_out_images folder<br>
<br>
2.Import libraries
```
import numpy as np
import pandas as pd
import scipy.io
import my_function as mf
from os import listdir
from os.path import isfile, join
from PIL import Image as PILImage
import fastai as fastai
from fastai.vision import *
from fastai.metrics import error_rate
```
3.Load image into **hold_out_image** folder<br>
<br>
4.Load the pretrained model
```
learn = load_learner('data/','export152_all_486.pkl')
```
5.Run this line to get all file names in **hold_out_image** folder<br>
```
# lists of file in hold_out_images folder
file_lists = [f for f in listdir('hold_out_images') if isfile(join('hold_out_images', f))]
file_lists.sort()
```
6. Run this code if you don't use or can't use gpu
```
# run this line if there is no gpu detected
#defaults.device = torch.device('cpu')
```
7.Run this code to initiate prediction
```
# run image prediction
ls_prd = []
df_conf=pd.DataFrame()
for pic_name in file_lists:
    img = open_image('hold_out_images/'+str(pic_name))
    pred,_,conf = learn.predict(img)
    ls_prd.append(str(pred))
    df_conf[str(pic_name)] = pd.Series(conf) 
```
8.Show prediction<br>
```
#Show category
ls_prd
```
```
#Show confident for each category
df_conf
```
