#!/usr/bin/env python
# coding: utf-8

# # Optimized Image Sampling for Explanations 

# In[590]:


import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from skimage.transform import resize
from tqdm import tqdm
from scipy import linalg
import time
import math
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# ## Change code below to incorporate your *model* and *input processing*

# ### Define your model here:

# In[591]:


from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
# from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions



from keras import backend as K

# from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions


# In[592]:


class Model():
    def __init__(self):
        K.set_learning_phase(0)
        self.model = ResNet50()
        self.input_size = (224, 224)
        
    def run_on_batch(self, x):
        return self.model.predict(x)


# ### Load and preprocess image

# In[593]:


from keras.preprocessing import image
import pandas as pd


# In[594]:


batch_size = 100
config = {}
config['learning_rate'] = 0.02
config['momentum'] = 0.1
img_name = "persian cat"


def load_img(path):
    img = image.load_img(path, target_size=model.input_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x



def gen_initial_mask( s, p1):
    
    grid = np.random.rand(s, s) > p1
    grid = grid.astype('float32')
    return grid

def upsize_mask(s, mask):
    
    cell_size = np.ceil(np.array(model.input_size) / s)
    up_size = (s + 1) * cell_size
    
    masks = np.empty((224, 224))

    # Random shifts
    x = np.random.randint(0, cell_size[0])
    y = np.random.randint(0, cell_size[1])
    # Linear upsampling and cropping
    masks[:, :] = resize(mask, up_size, order=1, mode='reflect',
            anti_aliasing=False)[x:x + model.input_size[0], y:y + model.input_size[1]]
    
    masks = masks.reshape(*model.input_size, 1)
    
    return masks


def do_plot(ax):
    ax.plot([1,2,3], [4,5,6], 'k.')

def normalize(m):
    m = np.maximum(m, 0)
    m_max = m.max() 
    if m_max != 0: 
        m = m / m_max
    return m


def explain_and_update(model, inp, N, class_idx, itr, img):
    preds = []
    s = 8
    # Make sure multiplication is being done for correct axes
    
    ct = 20
    cols = 3
    rows = int(math.ceil(ct / cols))

    
    ct = 0
    pred_prev = 0.0
    m = np.empty((inp.shape[1], inp.shape[1]))
    
    for i in range(0, N):
    
        if i == 0:
            mask = gen_initial_mask(s, 0.7)
            mask_ = upsize_mask(s, mask)

        masked_inp = mask_*inp
        masked = masked_inp.reshape(1, masked_inp.shape[1], masked_inp.shape[2], 
                                    masked_inp.shape[3])
        pred = model.run_on_batch(masked)
        pred_curr = pred[:,class_idx]
        delta_pred = pred_curr - pred_prev
        # current for next iteration
        mask_prev = np.copy(mask)
        obs_r_prev, obs_c_prev= np.where(mask > 0.0)
        obs_prev = pd.DataFrame([obs_r_prev, obs_c_prev])
        
        masked_r_prev, masked_c_prev = np.where(mask == 0.0)
        obs_prev = pd.DataFrame([obs_r_prev.T, obs_c_prev.T])
        
        # ith iteration input sample selection and mask update
        mask[obs_r_prev, obs_c_prev] = np.random.rand(len(obs_r_prev)) > (1-pred[:,class_idx])
        mask[masked_r_prev, masked_c_prev] = np.random.rand(len(masked_r_prev)) > pred[:,class_idx]
        
        delta_mask = np.subtract(mask , mask_prev)
        # regularizer
        if delta_pred < 0:
            r, c = np.where(delta_mask < 0)
            
            mask[r,c] = mask[r,c] - config['learning_rate']*delta_pred*mask[r,c]
            
            
            r, c = np.where(delta_mask > 0)
            
            mask[r,c] =  mask[r,c] + config['learning_rate']*delta_pred*mask[r,c]
            

        if delta_pred >= 0:
            r, c = np.where(delta_mask < 0)
            
            mask[r,c] = mask[r,c] + config['learning_rate']*delta_pred*mask[r,c]
            
            
            r, c = np.where(delta_mask > 0)
            
            mask[r,c] =  mask[r,c] - config['learning_rate']*delta_pred*mask[r,c]
            
    
        obs_r_curr, obs_c_curr= np.where(mask > 0.0) 
        obs_curr = pd.DataFrame([obs_r_curr, obs_c_curr])
        
        new_mask_ = upsize_mask(s, mask)
        mask_ = new_mask_
        m += new_mask_[:,:,0]
        
        if ((i==0) and ((np.mean(m) < 0) or (np.mean(m) >5) or np.isnan(np.mean(m)))):
            # del m
            print(np.mean(m))
            return None
        
        if ((i%100 == 0)):
            
            print(i, np.mean(m), np.std(m))
        pred_prev = np.copy(pred[:,class_idx])

        
        
    return m


# In[ ]:





# ---

# In[442]:


def class_name(idx):
    return decode_predictions(np.eye(1, 1000, idx))[0][0][1]


# ## Running explanations

# In[443]:





def process_sal(sal_oise):
    # check for pixels which are highlighted due to the mask in focus
    sal_oise = sal_oise - 0.96*sal_oise.mean()
#     sal_oise = (sal_oise + 1e-10) / (np.linalg.norm(sal_oise, 2) + 1e-10)
#     sal_oise = cv2.resize(sal_oise, (224, 224), cv2.INTER_LINEAR)
    sal_oise = np.maximum(sal_oise, 0)
    sal_oise = sal_oise / sal_oise.max()
    
    return sal_oise


N = 5000
itr = 1
model = Model()
exp_dir = '/rhome/s7thakur/RISE/imagenet_test/oise/resnet/'

for image_file_path in glob.glob('/rhome/s7thakur/RISE/imagenet_test/inputs/*.jpg'):

    print(image_file_path)
        
    img, x = load_img(image_file_path)
    preds = model.run_on_batch(x)
    class_idx = np.argmax(preds, axis=1)[0]
    preds = decode_predictions(preds, top=5)
    
    if exp_dir + os.path.splitext(os.path.basename(image_file_path))[0]+'_'+str(class_idx)+'_oise.npy' in glob.glob(exp_dir+'*.npy'):
        print(image_file_path + 'explanation exists')
        continue

    # hopefully, we can say that the prediction does in fact makes sense
    for i, (imagenetID, label, prob) in enumerate(preds[0]):
        print('{}. {}: {:.2f}%'.format(i + 1, label, prob * 100))


    sal_oise = explain_and_update(model, x, N, class_idx, itr, img)
    
    
    while(isinstance(sal_oise, type(None))):
        print('Attemting to explain again')
        time.sleep(10)
        sal_oise = explain_and_update(model, x, N, class_idx, itr, img)
        
    np.save(exp_dir+os.path.splitext(os.path.basename(image_file_path))[0]+ '_' + str(class_idx) +'_oise' +'.npy', sal_oise)


