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
    img = np.array(img)
    dimx = len(img)
    dimy = len(img[0])
    rx = int(dimx / bin_size) 
    ry = int(dimy / bin_size) 
   
    shape = (rx,bin_size,ry,bin_size)        
    return img.reshape(shape).mean(-1).mean(1)


def center(img,n_draw = 500_000, u_thresh = 1,l_thresh = 0.5):
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
    

#binned arrays are easier to use for center detection
#bin value must be a common divisor of original image dimensions
binsize = 4
#scl is half the size of your cropped centered image
scl = 300
#fetch is the number of images you wish to combine (per color channel)
fetch = int(input('how many images per color channel?\n'))
print('\n')



img_paths = ['Jupiter/B_filter/Light/2023-11-01_21_31_34Z/','Jupiter/R_filter/Brights/2023-11-01_21_35_05Z/','Jupiter/V_filter/Brights/2023-11-01_21_38_05Z/']
out_paths = ['Blues','Reds','Greens']
color_img = [[[0,0,0] for i in range(2 * scl)] for j in range(2 * scl)]
color_channels = [] 

for i,img_path in enumerate(img_paths):
    print('working on',out_paths[i].lower(),'...')
    names = [img_path + img_name for e,img_name in enumerate(os.listdir(img_path)) if fnmatch.fnmatch(img_name,'Light_*.fit') and e < fetch]
    print('fetched %s file(s)'%(len(names)))
    snimgs = []
    
    for e,name in enumerate(tqdm(names)):
        nimg = normalize(fits.getdata(name).data)
        cimg = compress(nimg,bin_size = binsize)
        ncimg = normalize(cimg)
        
        x,y,xs = center(ncimg)
        
        
        """ uncomment to check if center detection is picking up noise
        xs = np.array(xs)
        plt.scatter(xs[:,0], xs[:,1])
        plt.show()
        """
        
        x,y = int(x*binsize),int(y*binsize)

        snimg = nimg[x - scl:x + scl, y - scl:y+scl]
        snimgs += [snimg]
        
        del nimg

        
    snimgs = np.array(snimgs)
    master_snimgs = snimgs.mean(axis = 0)
    color_channels += [np.asarray(master_snimgs)]
    print('done with',out_paths[i].lower(),'\n')




for i in range(2 * scl):
    for j in range(2 * scl):
        color_img[i][j][0] = int(255 * color_channels[1][i][j] ** 2)
        color_img[i][j][1] = int(255 * color_channels[2][i][j] ** 2)
        color_img[i][j][2] = int(255 * color_channels[0][i][j] ** 2)
     

plt.imsave('img%s.png'%(fetch),np.uint8(np.array(color_img)))
print('color image saved as','img%s.png'%(fetch))









