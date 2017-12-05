
# coding: utf-8

# In[1]:

import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.filters import threshold_mean
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
from scipy.ndimage.measurements import center_of_mass
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage.filters import median_filter
from skimage.morphology import remove_small_objects


# In[ ]:

def prepro(janllsm, flag = 'S'):
    """
    test
    
    """
    janl = io.imread(janllsm)
    green = janl[:,:,:,0]
    red = janl[:,:,:,1]
    blue = janl[:,:,:,2]
    blue_CNS = np.empty(blue.shape)
    green_CNS = np.empty(blue.shape)
    CNS = np.zeros(blue.shape)
    for i,zslice in enumerate(red):
        CNS_mask = zslice > threshold_mean(zslice)
        blue_CNS[i] = CNS_mask * blue[i]
        green_CNS[i] = CNS_mask * green[i]
        CNS[i] = CNS_mask
    xcenter = center_of_mass(CNS)
    VNC = CNS[:,math.ceil(xcenter[1]):,math.ceil(xcenter[2]) - 300:math.ceil(xcenter[2]) + 300]
    VNC_blue = blue_CNS[:,math.ceil(xcenter[1]):,math.ceil(xcenter[2]) - 300:math.ceil(xcenter[2]) + 300]
    VNC_green = green_CNS[:,math.ceil(xcenter[1]):,math.ceil(xcenter[2]) - 300:math.ceil(xcenter[2]) + 300]
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=int(10e5), verbose=1).fit(VNC_blue.reshape(-1,1))
    center_values = kmeans.cluster_centers_
    sorted_center = np.argsort(center_values, 0)
    lut = np.arange(4)
    lut[sorted_center] = np.arange(4).astype(int)
    labeled = lut[kmeans.labels_].reshape(VNC_blue.shape)
    if flag == 'F':
        labeled = np.absolute(labeled//2 - 1)
        result = (labeled * VNC_green).astype('uint8')
        io.imsave('/home/yingtao/Desktop/annote3D/image.tif', result)
        return result, green[:,math.ceil(xcenter[1]):,math.ceil(xcenter[2]) - 300:math.ceil(xcenter[2]) + 300]
    else:
        median = median_filter(labeled.astype('int8'), 11)
        mask = median//2
        removed = remove_small_objects(mask, 50000)
        removed = np.absolute(removed - 1)
        result = (removed * VNC_green).astype('uint8')
        io.imsave('/home/yingtao/Desktop/annote3D/image.tif', result)
        return result, green[:,math.ceil(xcenter[1]):,math.ceil(xcenter[2]) - 300:math.ceil(xcenter[2]) + 300]

    



