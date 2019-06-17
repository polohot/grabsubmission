import numpy as np
import pandas as pd
import scipy.io
import my_function as mf
from os import listdir
from os.path import isfile, join

import fastai as fastai
from fastai.vision import *
from fastai.metrics import error_rate

'''
aa = learn34.get_preds(DatasetType.Test)
bb = learn34.get_preds(DatasetType.Train)


learn.data.train_dl
learn.data.valid_dl
learn.data.test_dl

learn.data.test_dl = data.train_dl 

preds, y = learn.get_preds(DatasetType.Test)
y = torch.argmax(preds, dim=1)

src = ImageList.from_df(path=path, df=df_in).split_none().label_from_df()
data = src.transform(tfms,size=sz).databunch(bs=bs) 
src = ImageList.from_df(path=path, df=df_in).split_by_rand_pct(val_pct).label_from_df()
data = src.transform(tfms,size=sz).databunch(bs=bs)  
'''

def make_tst_trn(df_in):
    """
    Returns a train and test dataframe by checking column 'test'
    """
    ls_trn_path = []
    ls_tst_path = []
    ls_trn_class = []
    ls_tst_class = []
    for i in range(0, len(df_in)):
        if df_in['test'][i][0][0] == 0:
            ls_trn_path.append(str(df_in['relative_im_path'][i][0]))
            ls_trn_class.append(df_in['class'][i][0][0])
        else:
            ls_tst_path.append(str(df_in['relative_im_path'][i][0]))
            ls_tst_class.append(df_in['class'][i][0][0])
    trn = pd.DataFrame({'name':ls_trn_path,'cat':ls_trn_class})
    tst = pd.DataFrame({'name':ls_tst_path,'cat':ls_tst_class})
    return trn,tst

def make_label(df_in,df_map):
    """
    return lists of labelled car name
    """
    ls_label=[]
    for i in range(0,len(df_in)):
        i_cat = df_in['cat'][i]
        ls_label.append(str(df_map[0][i_cat-1][0]))
    return ls_label

def transfer_learner(source,learn,tfms,sz,bs,cycle,lr=1e-3):
    data = source.transform(tfms,size=sz).databunch(bs=bs)
    learn.data=data
    learn.freeze()
    learn.fit_one_cycle(cycle,max_lr=lr)
    learn.unfreeze()
    learn.fit_one_cycle(cycle,max_lr=slice(lr/1000,lr/10))
    return learn

def calc_accuracy(learn,df_in):
    """
    Function to calculate accuracy from dataframes
    Somehow I tried, calculate picture one by one gives better precision
    """
    ls_prd=[]
    ls_real=[]
    df_conf=pd.DataFrame()
    for i in range(0,len(df_in)):
        img = open_image('data/'+df_in['name'][i])
        pred_class,_,conf = learn.predict(img) 
        ls_prd.append(str(pred_class))
        ls_real.append(df_in['label'][i])
        df_conf[str(i)] = pd.Series(conf)    
        if (i+1)%2000 == 0:
            score = pd.DataFrame({'prd':ls_prd,'act':ls_real})
            correct = np.where(score['prd'] == score['act'],1,0).sum() / len(score)    
            print(i+1,correct)
    score = pd.DataFrame({'prd':ls_prd,'act':ls_real})
    correct = np.where(score['prd'] == score['act'],1,0).sum() / len(score)  
    return correct

def load_df_tst():
    mat_anno = scipy.io.loadmat('data/cars_annos.mat')
    df_all = pd.DataFrame(np.hstack((mat_anno['annotations'])))
    df_map = pd.DataFrame(np.hstack((mat_anno['class_names'])))
    _,df_tst = make_tst_trn(df_all)
    df_tst['label'] = make_label(df_tst,df_map)
    df_tst = df_tst.reset_index(drop=True)
    df_tst = df_tst[['name','label']]
    return df_tst

def create_cropped_data_and_save_in_folder(df_in):    
    from imageai.Detection import ObjectDetection
    from PIL import Image as PILImage
    import os
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    for i in range(0,len(df_in)):
        input_path = 'data/' + df_in['name'][i]
        output_path = 'data/car_crop_demo/' + (df_in['name'][i][-10:])
        cat_i = df_in['name'][i]
        detections = detector.detectObjectsFromImage(input_image=input_path,                                             
                                                     output_image_path='data/temp.jpg',
                                                     extract_detected_objects = False,
                                                     display_percentage_probability=False,
                                                     display_object_name=False)
        df_det = pd.DataFrame(detections)            
        if len(df_det) != 0:     
            if len(df_det) == 1:
                bound = df_det['box_points'][0]
                x0,y0,x1,y1 = bound[0],bound[1],bound[2],bound[3]
                # PIL image crop
                im = PILImage.open(input_path)
                crop_box = (x0, y0, x1, y1)
                crop = im.crop(crop_box)
                crop.save(output_path)
            elif len(df_det) > 1:            
                ls_sz = []
                for j in range(0,len(df_det)):
                    bound_j = df_det['box_points'][j]
                    size_j = (bound_j[2] - bound_j[0]) * (bound_j[3] - bound_j[1])
                    ls_sz.append(size_j)
                df_det['sz'] = ls_sz            
                df_det = df_det.sort_values('sz',ascending=False).reset_index(drop=True)
                bound = df_det['box_points'][0]
                x0,y0,x1,y1 = bound[0],bound[1],bound[2],bound[3]
                # PIL image crop
                im = PILImage.open(input_path)
                crop_box = (x0, y0, x1, y1)
                crop = im.crop(crop_box)
                crop.save(output_path)
            else:
                im = PILImage.open(input_path)
                im.save(output_path)
        elif len(df_det) == 0:
            im = PILImage.open(input_path)
            im.save(output_path)
        print(i)


