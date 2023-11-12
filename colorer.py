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
    ret = ret / ret.max()
    
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
        
    return ret[0],ret[1]
    

#binned arrays are easier to use for center detection
#bin value must be a common divisor of original image dimensions
binsize = 4
#scl is half the size of your cropped centered image
scl = 300
#color channels you wish to include
colorstring =  input('include color(s) (ex. rgb): ')
#nfetch is the number of images you wish to combine (per color channel)
nfetch = int(input('no. images per color channel: '))
p = float(input('centering power (ideal ~6): '))
#maps color letter to color name
l2n = {'r':'reds','b':'blues','g':'greens'}
l2d = {'r':0,'g':1,'b':2}

colorkey = ''
img_paths = []
for letter in colorstring:
    if letter.lower() == 'r' and 'r' not in colorkey:
        colorkey += 'r'
        img_paths += ['Jupiter/R_filter/Brights/2023-11-01_21_35_05Z/']
    elif letter.lower() == 'g' and 'g' not in colorkey:
        colorkey += 'g'
        img_paths += ['Jupiter/V_filter/Brights/2023-11-01_21_38_05Z/']
    elif letter.lower() == 'b' and 'b' not in colorkey:
        colorkey += 'b'
        img_paths += ['Jupiter/B_filter/Light/2023-11-01_21_31_34Z/']

ncolor = len(colorkey)

print('\n')
ndraw = int(10**p)
color_img = [[[0,0,0] for i in range(2 * scl)] for j in range(2 * scl)]
color_channels = [] 


for i,img_path in enumerate(img_paths):
    print('working on',l2n[colorkey[i]]+'...')
    names = [img_path + img_name for e,img_name in enumerate(os.listdir(img_path)) if fnmatch.fnmatch(img_name,'Light_*.fit') and (e < nfetch or not nfetch)]
    print('fetched %s file(s)'%(len(names)))
    snimgs = []
    
    for e,name in enumerate(tqdm(names)):
        nimg = normalize(fits.getdata(name).data)
        cimg = compress(nimg,bin_size = binsize)
        ncimg = normalize(cimg)
        
        x,y = center(ncimg,n_draw = ndraw)
        x,y = int(x*binsize),int(y*binsize)

        snimg = nimg[x - scl:x + scl, y - scl:y+scl]
        snimgs += [snimg]
        
        
        del nimg,cimg,ncimg,snimg

        
    snimgs = np.array(snimgs)
    master_snimgs = snimgs.mean(axis = 0)
    color_channels += [np.asarray(master_snimgs)]
    print('done with',l2n[colorkey[i]],'\n')
    del snimgs, master_snimgs


for i in range(2 * scl):
    for j in range(2 * scl):
        color = [0,0,0]
        for c in range(ncolor):
            color[l2d[colorkey[c]]] += color_channels[c][i][j]
            
        for c in range(3):
            color_img[i][j][c] = int(255 * color[c] ** 2)
            
        del color
     

out_name = 'Pics/jupiter_bin%s_pow%s-%s_%s.png'%(nfetch,int(p),int(round(p%1,1) * 10),colorkey)
plt.imsave(out_name,np.uint8(np.array(color_img)))
print('color image saved as',out_name)









