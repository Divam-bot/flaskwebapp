import cv2


from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext','autoreload')
get_ipython().run_line_magic('autoreload','2')


np.set_printoptions(threshold=np.nan)


class VideoCamera(object):

    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    load_weights_from_FaceNet(FRmodel)
    
    database = {}
    
    database["divam"] = img_to_encoding("images/divam.jpg", FRmodel)
    database["shrizz"] = img_to_encoding("images/shrizz.jpg", FRmodel)
    
    def who_is_it(image_path, database, model):
        encoding = img_to_encoding(image_path , model)
        min_dist = 150
     
        for (name, db_enc) in database.items():
            dist = np.linalg.norm( encoding - db_enc )
            if dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist > 0.7:
            print("Not in the database.")
        else:
            print ("it's " + str(identity) + ", the distance is " + str(min_dist))
            
        return min_dist, identity
    
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()        

    def get_frame(self):
        ret, frame = self.video.read()
    
        # DO WHAT YOU WANT WITH TENSORFLOW / KERAS AND OPENCV
    
        ret, jpeg = cv2.imencode('.jpg', frame)
        who_is_it("jpeg", database, FRmodel)
        return jpeg.tobytes()