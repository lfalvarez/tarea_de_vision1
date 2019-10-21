'''
Created on Oct 19, 2019

@author: jsaavedr
'''

import hog_features
import pai
import data
import os
import argparse
import numpy as np

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-datadir', type = str, required = True)
    args = parser.parse_args()
    str_datadir = args.datadir
    str_file = os.path.join(str_datadir, "data.list")
    data_ = data.read_data(str_file)
    
    filenames, labels = zip(*data_)
    np_labels = np.array(labels, np.int)
    image_size = (128,128)
    grid_size = (8,8)
    orientations = 8
    hog = hog_features.HOG(image_size, orientations, grid_size)
    dim = hog.get_lenght()
    n_images = len(filenames)
    features = np.zeros((n_images, dim), np.float32)
    print('feat shape {}'.format(features.shape))
    for i, filename in enumerate(filenames) :
        filename = os.path.join(str_datadir,'png_w256', filename)
        image = pai.imread(filename, as_gray = True)    
        fv = hog.get_hog(image)        
        features[i,:] = fv
        if (i+1) % 100 == 0 :
            print('{0:5.2f} %'.format(((i+1)*100)/n_images))
    
    str_outfile = os.path.join(str_datadir, 'features.np')
    str_labels = os.path.join(str_datadir, 'labels.np')
    features.tofile(str_outfile)
    print('feature saved at {}'.format(str_outfile))
    
    np_labels.tofile(str_labels)
    print('labels saved at {}'.format(str_labels))