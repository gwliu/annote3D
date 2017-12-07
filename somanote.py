
import numpy as np
from math import log
import itertools
from skimage.feature import peak_local_max 
from scipy import ndimage
import skimage.io as io
from skimage import filters
import itertools as itt
import preprocessor
from math import log
from os import listdir
import subprocess
from os.path import isfile, join
import math


"""
Detect somata in janelia images

"""

mypath = '/home/yingtao/Desktop/annote3D/'
desktop = '/home/yingtao/Desktop/'


##old fast function to prune skimage detected result
def _prune_fast(blobs, overlap, ratio):
    newblobs = [p for i, p in enumerate(blobs) if all(_distance(k, p, ratio)>overlap for k in blobs[i+1:])]
    return newblobs



##old functions to prune skimage detected result
#Prune close points
def _distance(blob1, blob2, ratio):
    dif = np.array([(blob1[0]-blob2[0])*ratio, blob1[1]-blob2[1],  blob1[2]-blob2[2]])
    dis = np.linalg.norm(dif)
    return dis
def _prune_blobs(blobs, overlap, ratio):
    for blob1, blob2 in itt.combinations(blobs, 2):
        if _distance(blob1, blob2, ratio) < overlap:
            if blob1[3] > blob2[3]:
                blob2[3] = 0
            else:
                blob1[3] = 0
    return blobs

def getboxslice(image,center,xd,yd):
    center = center.astype('int')
    try:
        temp = image[center[0] - 1,
                     center[2]-xd//2:center[2]+xd//2+1, 
                     center[1]-yd//2:center[1]+yd//2+1,
                     ]
        if temp.size == xd*yd:
            return temp
        else:
            return 0 
    except IndexError:
        return 0
def _discenter(x,y,xd,yd):
    dis = (x - xd//2) ** 2 / (xd//2) ** 2  + (y - yd//2) ** 2 / (yd//2) ** 2
    if  0.2 < dis < 0.8:
        return 1
    else:
        return 0
#return a ellipse sphere kernal
def addvalue(dog_list, image):
    xd, yd = 21, 21
    temp = [_discenter(x, y, xd, yd) for x in range(xd) for y in range(yd)]
    coin_kernel = np.array(temp).reshape(xd,yd)
    pixelvalue = np.array([np.sum(coin_kernel * getboxslice(image, i, xd, yd)) for i in dog_list])
    return np.concatenate([dog_list,pixelvalue.reshape(-1,1)],axis=1)
def _distance(blob1, blob2):
    dif = np.array([blob1[1]-blob2[1],  blob1[2]-blob2[2]])
    dis = np.linalg.norm(dif)
    return dis
def _prune_blobs(blobs, overlap):
    for i, blob1 in enumerate(blobs):
        k = blob1[0]
        for blob2 in blobs[i:,]:
            if blob2[0] - k > 2.5:
                break
            if _distance(blob1, blob2) < overlap and 0 < abs(blob1[0] - blob2[0]) < 3:
                if blob1[3] > blob2[3]:
                    blob2[3] = 0
                else:
                    blob1[3] = 0
    blobs = np.array([blob for blob in blobs if blob[3] != 0])
    return blobs


#calculate dog, detect peaks
def dog_maxima(terminal, sigma, zx_ratio=4, blur_sigma=0.3, size = 8, prune_dis = 2):
    s = sigma
    gaussian_images0 = ndimage.gaussian_filter(terminal, (s/4, s,s))
    gaussian_images1 = ndimage.gaussian_filter(terminal, ((s*1.3)/4, s*1.3,s*1.3))
    dog_image = np.abs((gaussian_images1* 1.3 ** 3).astype('int16') - gaussian_images0.astype("int16"))
    dog_list = getmaxima(dog_image)
    
    #blurred = ndimage.filters.gaussian_filter(dog_image.astype("int16"), sigma=(blur_sigma/zx_ratio, blur_sigma, blur_sigma))
    #local_maxima = peak_local_max(dog_image, min_distance=5, threshold_rel=0.05, exclude_border=True, num_peaks_per_label==1000)#, threshold_abs=s, footprint=np.ones([size//zx_ratio + 1, size, size]))
    #print('a')
    #dog_list = [1,1]
    #dog_list = _prune_fast(local_maxima, prune_dis, zx_ratio)
    return dog_list, np.swapaxes(dog_image,1,2) #dog_list_pruned, dog_images, blurred


#detect peaks by imagej function

def getfiles(filepath):
    files = [join(filepath, f) for f in listdir(filepath) if isfile(join(filepath, f))]
    files.sort()
    return files
def getlsmfiles(filepath):
    files = [join(filepath, f) for f in listdir(filepath) if f[-3:] == 'lsm']
    files.sort()
    return files
def getmaxima(dog_image):
    io.imsave(mypath +'original_images/1106.tif', dog_image.astype('int16'))
    p = subprocess.Popen(['xvfb-run','-a', desktop + "Fiji.app/./ImageJ-linux64", "-macro",
                          mypath + "macros/findmaxima.ijm"])
    p.wait()
    onlyfiles = getfiles(mypath + "maximafile")
    maxima_data = np.empty([0,3])
    for i, files in enumerate(onlyfiles):
        temp = np.genfromtxt(files, delimiter=',')
        try:
            temp[:, 0] = i + 1
        except IndexError:
            continue           
        maxima_data = np.append(maxima_data, temp, axis = 0)
    dog_list = np.array([i for i in maxima_data if not math.isnan(i[1])])
    p = subprocess.Popen("rm "+ mypath + "maximafile/*", shell = True)
    #order of coordinate z,x,y
    return dog_list


#prepare sample
def samplefromindex(dog_list, positive_labels, image, xysize, sample_size, xz_ratio):
    samples = np.empty([0, sample_size])
    labels = []
    for i, coord in enumerate(dog_list):
        coord = coord.astype(int)
        if coord[0] < xysize//xz_ratio:
            coord[0] = xysize//xz_ratio
        elif coord[0] > image.shape[0] - xysize//xz_ratio - 0.5:
            coord[0] = image.shape[0] - xysize//xz_ratio - 1
        if coord[1] < xysize + 0.5:
            coord[1] = xysize
        elif coord[1] > image.shape[2] - xysize - 0.5:
            coord[1] = image.shape[2] - xysize - 1
        if coord[2]< xysize + 0.5:
            coord[2] = xysize
        elif coord[2] > image.shape[1] - xysize - 0.5:
            coord[2] = image.shape[1] - xysize - 1
        coord_slice = np.index_exp[coord[0]-xysize//xz_ratio:coord[0]+xysize//xz_ratio+1, 
                                      coord[2]-xysize:coord[2]+xysize+1,  coord[1]-xysize:coord[1]+xysize+1]
        temp = image[coord_slice].flatten()
        if temp.shape[0] == samples.shape[1]:
            temp = temp.reshape(1,-1)
            samples = np.append(samples, temp, axis=0)
        else:
            print(i)
            continue
        if i in positive_labels:
            labels.append(1)
        else:
            labels.append(0)
    return samples, labels



#batch process
def batch_label(path = 'default path'):
    if path == 'default path':
        path = '/home/yingtao/Desktop/annote3D/' + "janelia_gal4"
    labels = []
    plabels_indiv = []
    samples_indiv = []
    sample_xysize = 15
    xz_ratio = 4
    sample_size = (sample_xysize//xz_ratio * 2 + 1)*(sample_xysize*2 + 1)**2
    samples = np.empty([0, sample_size])
    janl_lsms = getlsmfiles(mypath + "janelia_gal4")
    stopflag = 0
    for i, janl_lsm in enumerate(janl_lsms):
        if stopflag == 1:
            break
        image, origin_image = preprocessor.prepro(janl_lsm)
        dog_list, dog_image = dog_maxima(image, 2, size =7)
        dog_list = addvalue(dog_list, origin_image)
        dog_list = _prune_blobs(dog_list, 10)
        np.savetxt(mypath + 'dog_maxima', dog_list)
        io.imsave(mypath + 'origin.tif', origin_image.astype("int16"))
        p = subprocess.Popen([desktop + "Fiji.app/./ImageJ-linux64", "-macro",
                          mypath + "macros/showcirc.ijm"])
        for temp in dog_list:
            s = input("What's the index of circled neuron? ")
            if s == 'e':
                print('next image')
                samples_indiv, labels_indiv = samplefromindex(dog_list, plabels_indiv, 
                                                                          origin_image, sample_xysize, sample_size, xz_ratio)
                samples = np.append(samples, samples_indiv, axis = 0)
                labels.extend(labels_indiv)
                plabels_indiv =[]
                np.savetxt(mypath + 'tmp/' + str(i) + 'sample', samples_indiv)
                np.savetxt(mypath + 'tmp/' + str(i) + 'labels', labels_indiv)
                break
            if s == 'd':
                print('deleteone')
                plabels_indiv.pop()
            if s == 'n':
                print('neglect it')
                break
            if s == 'stop':
                print('next image')
                samples_indiv, labels_indiv = samplefromindex(dog_list, plabels_indiv, 
                                                                          origin_image, sample_xysize, sample_size, xz_ratio)
                samples = np.append(samples, samples_indiv, axis = 0)
                labels.extend(labels_indiv)
                plabels_indiv =[]
                stopflag = 1
                np.savetxt(mypath + 'tmp/' + str(i) + 'samples', samples_indiv)
                np.savetxt(mypath + 'tmp/' + str(i) + 'labels', labels_indiv)
                break
            try:
                plabels_indiv.append(int(s))
            except ValueError:
                print('do it again!')
            print("neuron + " +s+"")
        np.savetxt(mypath + 'labeled_samples_file/' + str(i) + 'samples.txt', samples_indiv)
        np.savetxt(mypath + 'labeled_samples_file/' + str(i) + 'labels.txt', labels_indiv)
    return samples, labels

def batch_test(path = 'default path'):
    if path == 'default path':
        path = '/home/yingtao/Desktop/annote3D/' + "janelia_gal4"
    labels = []
    plabels_indiv = []
    samples_indiv = []
    sample_xysize = 15
    xz_ratio = 4
    sample_size = (sample_xysize//xz_ratio * 2 + 1)*(sample_xysize*2 + 1)**2
    samples = []
    janl_lsms = getlsmfiles(mypath + "janelia_gal4")
    stopflag = 0
    for i, janl_lsm in enumerate(janl_lsms):
        if stopflag == 1:
            break
        if i == 0:
            print('next')
            continue
        image, origin_image = preprocessor.prepro(janl_lsm)
        dog_list, dog_image = dog_maxima(image, 2, size =7)
        dog_list = addvalue(dog_list, origin_image)
        dog_list = _prune_blobs(dog_list, 10)
        plabels_indiv = np.ones(dog_list.shape[0])
        samples_indiv, labels_indiv = samplefromindex(dog_list, plabels_indiv, 
                                                                          origin_image, sample_xysize, sample_size, xz_ratio)
        samples.append(samples_indiv)
        np.savetxt(mypath + 'test_files/' + str(i) + 'samples.txt', samples_indiv)
        s = input("Do you want to stop?(y/n)")
        if s == 'y' or 'Y':
            break
    return samples