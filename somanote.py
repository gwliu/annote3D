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
import signal


"""
Detect somata in janelia images

"""

outputpath = '/home/yingtao/Desktop/annote3D/'
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
    xd, yd = 19, 19
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
def dog_maxima(terminal, sigma, zx_ratio=4, blur_sigma=0.3, prune_dis = 2):
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
def gettiffiles(filepath):
    files = [join(filepath, f) for f in listdir(filepath) if f[-3:] == 'tif']
    files.sort()
    return files
def getmaxima(dog_image):
    io.imsave(outputpath +'tmp/dog.tif', dog_image.astype('int16'))
    p = subprocess.Popen(['xvfb-run','-a', desktop + "Fiji.app/./ImageJ-linux64", "-macro",
                          mypath + "macros/findmaxima.ijm"])
    p.wait()
    onlyfiles = getfiles(outputpath +  "tmp/maximafiles")
    maxima_data = np.empty([0,3])
    for i, files in enumerate(onlyfiles):
        temp = np.genfromtxt(files, delimiter=',')
        try:
            temp[:, 0] = i + 1
        except IndexError:
            continue           
        maxima_data = np.append(maxima_data, temp, axis = 0)
    dog_list = np.array([i for i in maxima_data if not math.isnan(i[1])])
    p = subprocess.Popen("rm "+ outputpath + "tmp/maximafiles/*", shell = True)
    #order of coordinate z,x,y
    return dog_list

#prepare sample
def samplefromindex(dog_list, positive_labels, unknown_labels, image, blue, red, xysize, sample_size, xz_ratio):
    samples = np.empty([0, sample_size * 3])
    coords = []
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
        temp1 = blue[coord_slice].flatten()
        temp2 = red[coord_slice].flatten()
        if temp.shape[0] == sample_size:
            temp = temp.reshape(1,-1)
            temp1 = temp1.reshape(1,-1)
            temp2 = temp2.reshape(1,-1)
            sample = np.concatenate([temp,temp1,temp2], axis = 1)
            samples = np.append(samples,sample, axis=0)
        else:
            print(i)
            continue
        coords.append(coord)
        if i in positive_labels:
            labels.append(1)
        elif i in unknown_labels:
            labels.append(2)
        else:
            labels.append(0)
    return samples, labels, coords

def batch_label(inputpath = 'default path'):
    if inputpath == 'default path':
        inputpath = '/home/yingtao/Desktop/annote3D/' + "janelia_gal4"
    print(inputpath)
    labels = []
    plabels_indiv = []
    ulabels_indiv = []
    samples_indiv = []
    coords_indiv = []
    coords = []
    sample_xysize = 15
    xz_ratio = 4
    sample_size = (sample_xysize//xz_ratio * 2 + 1)*(sample_xysize*2 + 1)**2
    samples = np.empty([0, sample_size*3])
    janl_lsms = gettiffiles(inputpath)
    stopflag = 0
    nextimage = 0
    for i, janl_lsm in enumerate(janl_lsms):
        if stopflag == 1:
            break
        print(janl_lsm)
        flag, image, origin_image, blue, red = preprocessor.prepro(janl_lsm)
        if flag == 0:
            break
        dog_list, dog_image = dog_maxima(image, sigma = 2.5)
        dog_list = addvalue(dog_list, origin_image)
        dog_list = _prune_blobs(dog_list, 10)
        np.savetxt(outputpath + 'tmp/dog_maxima', dog_list)
        io.imsave(outputpath + 'tmp/origin.tif', origin_image.astype("uint8"))
        p = subprocess.Popen([desktop + "Fiji.app/./ImageJ-linux64", "-macro",
                          mypath + "macros/showcirc.ijm"])
        for temp in dog_list:
            s = input("What's the index of circled neuron?")
            if s == 'e':
                for temp2 in dog_list:
                    s = input("What's the index of unknown case?")
                    if s == 'e':
                        print('next image')
                        samples_indiv, labels_indiv, coords_indiv = samplefromindex(dog_list, plabels_indiv,
                                                                                    ulabels_indiv, origin_image, blue,
                                                                                    red, sample_xysize, sample_size,
                                                                                    xz_ratio)
                        samples = np.append(samples, samples_indiv, axis = 0)
                        coords.extend(coords_indiv)
                        labels.extend(labels_indiv)
                        plabels_indiv =[]
                        ulabels_indiv = []
                        np.savetxt(outputpath + 'tmp/' + str(i) + 'sample', samples_indiv)
                        np.savetxt(outputpath + 'tmp/' + str(i) + 'labels', labels_indiv)
                        nextimage = 1
                        break
                    if s == 'd':
                        print('deleteone')
                        ulabels_indiv.pop()
                    if s == 'stop':
                        print('stop')
                        samples_indiv, labels_indiv, coords_indiv = samplefromindex(dog_list, plabels_indiv,
                                                                                    ulabels_indiv, origin_image, blue,
                                                                                    red, sample_xysize, sample_size,
                                                                                    xz_ratio)
                        samples = np.append(samples, samples_indiv, axis = 0)
                        coords.extend(coords_indiv)
                        labels.extend(labels_indiv)
                        plabels_indiv =[]
                        ulabels_indiv = []
                        stopflag = 1
                        np.savetxt(outputpath + 'tmp/' + str(i) + 'samples', samples_indiv)
                        np.savetxt(outputpath + 'tmp/' + str(i) + 'labels', labels_indiv)
                        nextimage = 1
                        break
                    try:
                        ulabels_indiv.append(int(s))
                    except ValueError:
                        print('do it again!')
                    print("unknown + " +s+"")
            if nextimage == 1:
                nextimage = 0
                break
            if s == 'd':
                print('deleteone')
                plabels_indiv.pop()
            if s == 'n':
                print('neglect it')
                break
            try:
                plabels_indiv.append(int(s))
            except ValueError:
                print('do it againnnn!')
            print("neuron + " +s+"")
        np.savetxt(outputpath + 'labeled_samples_file/' + str(i) + 'samples.txt', samples_indiv)
        np.savetxt(outputpath + 'labeled_samples_file/' + str(i) + 'labels.txt', labels_indiv)
    return samples, labels, coords


def alarm_handler(signum, frame):
    raise TimeoutExpired

def input_with_timeout(prompt, timeout):
    # set signal handler
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(timeout) # produce SIGALRM in `timeout` seconds

    try:
        return input(prompt)
    finally:
        signal.alarm(0) # cancel alarm
def batch_test(inputpath = 'default path'):
    if inputpath == 'default path':
        inputpath = '/home/yingtao/Desktop/annote3D/' + "janelia_gal4"
    labels = []
    plabels_indiv = []
    samples_indiv = []
    ulabels_indiv = []
    sample_xysize = 15
    xz_ratio = 4
    sample_size = (sample_xysize//xz_ratio * 2 + 1)*(sample_xysize*2 + 1)**2
    samples = []
    janl_lsms = gettiffiles(inputpath)
    stopflag = 0
    s = 'n'
    for i, janl_lsm in enumerate(janl_lsms):
        if stopflag == 1:
            break
        if i == 0:
            print('next')
            continue
        flag, image, origin_image, blue, red = preprocessor.prepro(janl_lsm)
        if flag == 0:
            print('wrong image size')
            continue
        dog_list, dog_image = dog_maxima(image, sigma = 2.5)
        dog_list = addvalue(dog_list, origin_image)
        dog_list = _prune_blobs(dog_list, 10)
        samples_indiv, labels_indiv, coords_indiv = samplefromindex(dog_list, plabels_indiv, 
                                                                    ulabels_indiv, origin_image, blue, red, 
                                                                    sample_xysize, sample_size, xz_ratio)
        janl_file = janl_lsm[-23:-4]
        np.savetxt('/home/yingtao/Desktop/annote3D/tmp/samples' + janl_file + '.txt', samples_indiv)
        np.savetxt('/home/yingtao/Desktop/annote3D/tmp/coords' + janl_file + '.txt', coords_indiv)
        samples.append(samples_indiv)
        np.savetxt(outputpath + 'test_files/' + str(i) + 'samples.txt', samples_indiv)
        #s = input_with_timeout("Do you want to stop?(y/n)", 10)
        #if s == 'y' or 'Y':
        #    print('stop')
        #    break
    return samples