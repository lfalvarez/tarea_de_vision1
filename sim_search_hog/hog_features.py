'''
Created on Oct 19, 2019

@author: jsaavedr
'''

import skimage.feature as feat
import skimage.transform as trans
import pai
import sys
import numpy as np

class HOG :
    def __init__(self, image_size= (128,128), orientations = 8, grid_size = (4,4)):
        self.image_size = image_size
        self.orientations = orientations #number of orientations
        self.grid_size = grid_size  # numberl of blocks
        
    def get_lenght(self):
        return self.orientations * self.grid_size[0] * self.grid_size[1]

    def get_hog(self, image):
        image = trans.resize(image, self.image_size)
        image = pai.to_uint8(image)    
        fd = feat.hog(image, orientations= self.orientations,
                       pixels_per_cell=(self.image_size[0]/self.grid_size[0], 
                                        self.image_size[1]/self.grid_size[1]), 
                       cells_per_block=(1, 1), feature_vector = True)
        #normalizing the feature vector
        fd = np.sqrt(fd)
        norm = np.sqrt(np.dot(fd, fd));
        fd = fd / norm  
        return fd

#
if __name__ == '__main__' : 
    filename = sys.argv[1]
    image = pai.imread(filename, as_gray = True)
    hog = HOG()         
    h = hog.get_hog(image)
    norm = np.sqrt(np.dot(h, h));
    print(h)
    print(norm)
    print(h.shape)