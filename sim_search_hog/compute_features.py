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


class Config():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_size = (128, 128)
        self.grid_size = (8, 8)
        self.orientations = 8
        str_file = os.path.join(str_datadir, "{}.list".format(file_name))
        the_files_ = data.read_data(str_file)
        self.filenames, self.labels = zip(*the_files_)
        self.features = np.zeros((len(self.filenames), dim), np.float32)
        return features


def calculate_distance_to_all_images(str_datadir, original_hogs, original_labels):
    str_file = os.path.join(str_datadir, "test.list")
    test_files_ = data.read_data(str_file)
    filenames, labels = zip(*test_files_)

    image_size = (128, 128)
    grid_size = (8, 8)
    orientations = 8
    hog = hog_features.HOG(image_size, orientations, grid_size)
    dim = hog.get_lenght()
    n_images = len(filenames)
    features = np.zeros((n_images, dim), np.float32)
    print('feat shape {}'.format(features.shape))
    sum_of_aps = 0
    for i, (filename, current_label) in enumerate(zip(filenames, labels)) :
        filename = os.path.join(str_datadir,'png_w256', filename)
        image = pai.imread(filename, as_gray = True)
        this_hog = hog.get_hog(image)
        distances = []
        for index, (other_hog, other_label) in enumerate(zip(original_hogs, original_labels)):
            dist = np.linalg.norm(this_hog - other_hog)
            is_relevant = current_label == other_label ## Es relevante SSI el label del test es igual a la de la imagen cercana.
            distances.append({'is_relevant': is_relevant, 'dist': dist})

        sorted_distances = sorted(distances, key=lambda d: d['dist'])
        distances = sorted_distances[:25]
        relevant_counter = 0
        AP = 0
        for index, d in enumerate(distances):
            dividendo = 0
            if d['is_relevant']:
                relevant_counter += 1
                dividendo = relevant_counter
            AP += dividendo / (index + 1)
        if relevant_counter:
            AP = AP/relevant_counter
            sum_of_aps += AP

        '''if relevant_counter:
            rep = '('
            counter = 0
            for i, d in enumerate(distances):
                if d['is_relevant']:
                    counter += 1
                    rep += '{}/{} + '.format(counter, i+1)
            rep += '0)/{}'.format(relevant_counter)
            print(rep)
            print(a_p)'''
        ## frente a la consulta aquí imprimo la cantidad de imagenes relevantes por cada test!
        #print(a_p)
    mAP = (sum_of_aps/ len(filenames))
    print(mAP)


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
    
    str_outfile = os.path.join(str_datadir, 'features.np')
    str_labels = os.path.join(str_datadir, 'labels.np')
    features.tofile(str_outfile)
    print('feature saved at {}'.format(str_outfile))
    
    np_labels.tofile(str_labels)
    print('labels saved at {}'.format(str_labels))
    calculate_distance_to_all_images(str_datadir, features, labels)