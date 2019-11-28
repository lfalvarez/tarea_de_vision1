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
from PIL import Image



def calculate_distance_to_all_images(str_datadir, original_hogs, original_labels, original_filenames):
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
        for index, (other_hog, other_label, other_filename) in enumerate(zip(original_hogs, original_labels, original_filenames)):
            dist = np.linalg.norm(this_hog - other_hog)
            is_relevant = current_label == other_label ## Es relevante SSI el label del test es igual a la de la imagen cercana.
            datos_de_comparacion = {'is_relevant': is_relevant, 'dist': dist}
            datos_de_comparacion['other_filename'] = other_filename
            distances.append(datos_de_comparacion)

        sorted_distances = sorted(distances, key=lambda d: d['dist'])
        distances = sorted_distances[:10]
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

        if relevant_counter > 2 and relevant_counter < 5:
            para_mostrar = [filename]
            for d in distances[:10]:
                para_mostrar.append(os.path.join(str_datadir,'png_w256', d['other_filename']))
            final_dim = (256 * len(para_mostrar), 256)
            new_im = Image.new('L', final_dim)
            for j, im_file in enumerate(para_mostrar):
                im = Image.open(im_file)
                new_im.paste(im, (j*256, 0))
            new_im.save('{}_comparado.jpg'.format(i))
            print('{}_comparado.jpg'.format(i))



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
    calculate_distance_to_all_images(str_datadir, features, labels, filenames)