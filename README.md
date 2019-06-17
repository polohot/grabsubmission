Competitor : Khunakorn Luyaphan<br>
Email : gatekhunakorn@gmail.com
# Grab Submission (Test accuracy 92.2%)
This model is trained on stanford car dataset<br>
https://ai.stanford.edu/~jkrause/cars/car_dataset.html<br>
# Main notebook<br>
<a href="https://github.com/polohot/grabsubmission/blob/master/grab_challange/GRAB%20Image%20classification%20challange.ipynb">grab_challange/GRAB Image classification challange.ipynb</a><br>
# Environment Setup (Tested on windows)
**Please follow step by step**<br><br>
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
# Make prediction of test set<br>
**Section 3.2 of the notebook**<br>
1. Download trained model https://drive.google.com/open?id=1bXHQwkWT5fLlCq0Uye8G-kF7RoKTlI4q <br>
2. paste in directory **grabchallange/data/***
3. Download test data (car_ims) from <br>
http://imagenet.stanford.edu/internal/car196/cars_annos.mat
4. Unzip to **grabsubmission/data/car_ims/***
5. Import library at the top of the notebook
6. Go to section 3.2 of notebook, follow line by line
# Make prediction of hold_out set <br>
**Section 3.3 of the notebook**<br>
1. Download trained model https://drive.google.com/open?id=1bXHQwkWT5fLlCq0Uye8G-kF7RoKTlI4q <br>
2. paste in directory **grabchallange/data/***<br>
3. Copy pictures into **grabchallange/hold_out_images/*** folder
4. Import library at the top of the notebook
5. Go to section 3.3 of notebook, follow line by line
