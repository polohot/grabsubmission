# Grabsubmission (Test accuracy 92.2%)
# Environment Setup<br>
please use the environment from **environment.yml**<br>
# Pretrained Model Setup<br>
please download the pretrained model in this link <br>
https://drive.google.com/open?id=1jTnColYHZhnuckgop06OAzk8D-mROrG7 <br>
paste the file in this directory after clone <br> 
grabchallange/data/*
# Main notebook<br>
grab_challange/GRAB Image classification challange.ipynb
# Make prediction
Last section of the notebook states on how to make prediction on the new images using the model above<br>
1. copy pictures into hold_out_images folder
2. import libraries

# Import library dependencies
import numpy as np
import pandas as pd
import scipy.io
import my_function as mf
from os import listdir
from os.path import isfile, join
from PIL import Image as PILImage
# Import Fast.ai Library
import fastai as fastai
from fastai.vision import *
from fastai.metrics import error_rate
pd.options.mode.chained_assignment = None
