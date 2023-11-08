# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 01:09:49 2023

@author: lcuev
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import fnmatch
import os
from tqdm import tqdm


def normalize(img):
    ret = img - np.median(img)
    ret[ret<0] = 0
    den = ret.max()
    ret = ret / den
    
    return ret

                
def compress(img,bin_size):
    dimx = len(img)
    dimy = len(img[0])
    rx = int(dimx / bin_size) 
    ry = int(dimy / bin_size) 
    ret = np.zeros((rx,ry))
 
    for _x in range(rx):
        for _y in range(ry):
            x = _x * bin_size
            y = _y * bin_size
            if x + bin_size < dimx and y + bin_size <dimy :
                ret[_x,_y] = np.mean(img[x:x + bin_size, y : y + bin_size])
                
    return ret


def center(img,n_draw = 1000000, u_thresh = 0.9,l_thresh = 0.4):
    xs = []

    
    for draw in range(n_draw):
        rx = int(np.random.random() * len(img))
        ry = int(np.random.random() * len(img[0]))
        
        if u_thresh > img[rx,ry] > l_thresh:
            xs += [[rx,ry]]
    
    ret = [0,0] 
    
    for j in range(2):
        for x in xs:
                ret[j] += x[j] 
                
    for j in range(2):
        ret[j] /= len(xs)
        
    return ret[0],ret[1],xs
    
        
binsize = 20
scl = 300
fetch = 150



img_paths = ['Jupiter/B_filter/Light/2023-11-01_21_31_34Z/','Jupiter/R_filter/Brights/2023-11-01_21_35_05Z/','Jupiter/V_filter/Brights/2023-11-01_21_38_05Z/']
out_paths = ['Blues','Reds','Greens']
color_img = [[[0,0,0] for i in range(2 * scl)] for j in range(2 * scl)]
color_channels = [] 

for i,img_path in enumerate(img_paths):
    names = [img_path + img_name for e,img_name in enumerate(os.listdir(img_path)) if fnmatch.fnmatch(img_name,'Light_*.fit') and e < fetch]
    print('fetched %s file(s)'%(len(names)))
    snimgs = []
    
    for e,name in enumerate(tqdm(names)):
        nimg = normalize(fits.getdata(name).data)
        cimg = compress(nimg,bin_size = binsize)
        ncimg = normalize(cimg)
        
        x,y,xs = center(ncimg)
        x,y = int(x*binsize),int(y*binsize)

        snimg = nimg[x - scl:x + scl, y - scl:y+scl]
        snimgs += [snimg]
        
        del nimg

        
    
    
    
    snimgs = np.array(snimgs)
    master_snimgs = snimgs.mean(axis = 0)
    color_channels += [np.asarray(master_snimgs)]
    print('done with',out_paths[i].lower())




for i in range(2 * scl):
    for j in range(2 * scl):
        color_img[i][j][0] = int(255 * color_channels[1][i][j] ** 2)
        color_img[i][j][1] = int(255 * color_channels[2][i][j] ** 2)
        color_img[i][j][2] = int(255 * color_channels[0][i][j] ** 2)
     

plt.imsave('img%s.png'%(fetch),np.uint8(np.array(color_img)))
print('color image saved as','img%s.png'%(fetch))









