
# coding: utf-8

# System modules
import os
import warnings
import pickle
import glob
import sys
import logging
import datetime as dt
import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 

# External modules
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

# Internal modules
import pyclamster
from pyclamster.clustering.preprocess import LCN
from pyclamster.functions import rbDetection
from pyclamster.positioning import Projection

plt.style.use('ggplot')

warnings.catch_warnings()
warnings.filterwarnings('ignore')

#logging.basicConfig(level=logging.INFO)


# In[2]:

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
#image_directory = os.path.join(BASE_DIR, "examples", "images", "lex")
image_directory = '/home/tfinn/Data/Cloud_camera/lex/'
trained_models = os.path.join(BASE_DIR, "data")
plot_dir = os.path.join(BASE_DIR, 'plots')

debug = False


# In[3]:

predictor = pickle.load(open(os.path.join(trained_models, "kmeans.pk"), "rb"))

cams = []
cams.append(pickle.load(open(os.path.join(trained_models, 'sessions', 'FE3_session_new_600.pk'), mode='rb')))
cams.append(pickle.load(open(os.path.join(trained_models, 'sessions', 'FE4_session_new_600.pk'), mode='rb')))
cams[0].set_images(os.path.join(image_directory, 'cam3'))
cams[1].set_images(os.path.join(image_directory, 'cam4'))
matching = pyclamster.matching.Matching(greyscale=True)
dist = np.sqrt((cams[0].position.x-cams[1].position.x)**2+(cams[0].position.y-cams[1].position.y)**2)

height = {}


# In[4]:

# Initiate the multiprocessing pool
num_cores = 2
pool = ThreadPool(num_cores)


# In[5]:

def generate_doppel(images):
    start_time = time.time()
    k = 0
    clouds = []
    height[images[0].time] = []
    for img in images:
        image_lcn = pyclamster.Image(img)
        image_lcn.data = LCN(size=(50, 50, 3), scale=False).fit_transform(
            image_lcn.data / 256)
        w, h, _ = original_shape = image_lcn.data.shape
        raw_image_lcn = rbDetection(image_lcn.data).reshape((w * h, -1))
        label = predictor.predict(raw_image_lcn)
        label.reshape((w, h), replace=True)
        masks = label.getMaskStore()
        cloud_mask_num = [0] # cloud - sky choose right number (0 or 1)
        masks.denoise(cloud_mask_num, 5000)
        #cloud_labels_object, numLabels = masks.labelMask(cloud_mask_num)
        #cloud_labels_object, numLabels = masks.wsMask(cloud_mask_num, np.ones((14,14)))
        cloud_labels_object, numLabels = masks.regionMask(cloud_mask_num, (50, 50))
        cloud_store = cloud_labels_object.getMaskStore()
        cloud_lables = np.unique(cloud_labels_object.labels)[1:]
        clouds.append([cloud_store.getCloud(img, [k, ]) for k in cloud_lables])
        if debug:
            j = 0
            scipy.misc.imsave(
                os.path.join(plot_dir, "lables_kmean_{0:s}_{1:d}.png".format(img.time.strftime('%Y%m%d%H%M'), k)),
                             label.labels)
            scipy.misc.imsave(
                os.path.join(plot_dir, "lables_used_{0:s}_{1:d}.png".format(img.time.strftime('%Y%m%d%H%M'), k)),
                             cloud_labels_object.labels)
            print('finished image {0:s} of camera {1:d}'.format(img.time.strftime('%Y%m%d%H%M'), k))
        k += 1
    t=0
    if not(not clouds[0] or not clouds[1]):
        matching_result, _ = matching.matching(clouds[0], clouds[1], min_match_prob=0.95)
        for result in matching_result:
            spatial_cloud = result[1]
            successful = spatial_cloud.oper_mode()
            if debug and successful:
                fig = plt.figure()
                ax = plt.subplot(1,3,1)
                ax.axis('off')
                ax.imshow(result[1].clouds[0].image.data)
                ax = plt.subplot(1,3,2)
                ax.axis('off')
                ax.imshow(result[0].prob_map, interpolation='nearest')
                ax = plt.subplot(1,3,3)
                ax.axis('off')
                ax.imshow(result[1].clouds[1].image.data)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 'matching_{0:s}_{1:d}.png'.format(img.time.strftime('%Y%m%d%H%M'), t)))
                fig = plt.figure()
                ax = plt.subplot(1,3,1)
                ax.axis('off')
                ax.imshow(spatial_cloud.clouds[0].image.data)
                ax = plt.subplot(1,3,2)
                ax.axis('off')
                z = spatial_cloud.positions.z
                X, Y = np.meshgrid(range(z.shape[1]), range(z.shape[0]))
                CS = ax.contour(X, Y, z)
                manual_locations = [(-1, -1.4), (-0.62, -0.7), (-2, 0.5), (1.7, 1.2), (2.0, 1.4), (2.4, 1.7)]
                plt.clabel(CS, inline=1, fontsize=10)
                ax = plt.subplot(1,3,3)
                ax.axis('off')
                ax.imshow(spatial_cloud.clouds[1].image.data)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 'height{0:s}_{1:d}.png'.format(img.time.strftime('%Y%m%d%H%M'), t)))
            height[images[0].time].append(spatial_cloud.height)
            t+=1
    try:      
        print('finished image {0:s} calculations with {1:d} heights (min: {2:.1f}, max: {3:.1f}), duration: {4:.1f} s'.format(images[0].time.strftime('%Y%m%d%H%M'), t, np.nanmin(height[images[0].time]), np.nanmax(height[images[0].time]), time.time()-start_time))
    except:
        print('finished image {0:s} calculations with {1:d} heights, duration: {2:.1f} s'.format(images[0].time.strftime('%Y%m%d%H%M'), t, time.time()-start_time))


# In[ ]:

images_available = True
gens = [cams[0].iterate_over_rectified_images(), cams[1].iterate_over_rectified_images()]
images = [next(gens[0]), next(gens[1])]
while images_available:
    available_images = []
    while len(available_images)<num_cores:
        for k, img in enumerate(images):
            img.loadTimefromfilename('FE{0:d}_Image_%Y%m%d_%H%M%S_UTCp1.jpg'.format(k+3))
        if (images[0].time-images[1].time).seconds<60:
            available_images.append(images)
            try:
                images = [next(gens[0]), next(gens[1])]
            except:
                images_available = False
                cnt_thread = 99
        elif images[0].time<images[1].time:
            try:
                images[0] = next(gens[0])
            except:
                images_available = False
                cnt_thread = 99
        elif images[0].time>images[1].time:
            try:
                images[1] = next(gens[1])
            except:
                images_available = False
                cnt_thread = 99
    try:
        print('Found 8 image pairs, starting the multiprocessing.')
        pool.map(generate_doppel, available_images)
        pool.close() 
        pool.join()
    except Exception as e:
        print(e)
print('finished image processing')


# In[11]:

max_len = np.max([len(height[i]) for i in height.keys()])
df_height = pd.DataFrame(columns=range(max_len), index=height.keys())
for i in df_height.index:
    for k, j in enumerate(height[i]):
        if not j is None:
            if 70<j<15000:
                df_height.ix[i, k] = j
df_height = df_height.sort()
df_height.to_json(os.path.join(trained_models, 'heights_201609010900_4h_300_new.json'))


# In[ ]:



