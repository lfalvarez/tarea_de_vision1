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
        distances, filename = distancias_ordenadas_a_imagen_de_test(current_label, filename, hog, original_filenames, original_hogs,
                                                   original_labels, str_datadir)
        ## Si ya tengo las distancias de este ejemplo de test a todos los otros ejemplos
        ## Entonces puedo calcular su AP
        relevant_counter = 0
        AP = 0
        for index, d in enumerate(distances):
            dividendo = 0
            if d['is_relevant']:
                ## Si es relevante entonces el número de arriba de la división es la cantidad de aciertos
                relevant_counter += 1
                dividendo = relevant_counter
            ## el númnero de abajo siempre es el índice de donde estamos.
            AP += dividendo / (index + 1)
        if relevant_counter:
            ## La suma de todos los AP se divide por la cantidad total de
            ## elementos relevantes.
            AP = AP/relevant_counter
            sum_of_aps += AP

        imprimir_imagenes(distances, filename, i, relevant_counter, str_datadir)
    mAP = (sum_of_aps/ len(filenames))
    print(mAP)


def distancias_ordenadas_a_imagen_de_test(current_label, filename, hog, original_filenames, original_hogs, original_labels, str_datadir):
    filename = os.path.join(str_datadir, 'png_w256', filename)
    image = pai.imread(filename, as_gray=True)
    this_hog = hog.get_hog(image)
    distances = []
    # Por cada una de las imagenes de training (other_hogs)
    # calcularé la distancia euclidiana y si es relevante o no.
    for index, (other_hog, other_label, other_filename) in enumerate(zip(original_hogs, original_labels, original_filenames)):
        dist = np.linalg.norm(
            this_hog - other_hog)  ## calculando la distancia euclidiana entre el hog de la imagen de test y la previamente calculada
        is_relevant = current_label == other_label  ## Es relevante SSI el label del test es igual a la de la imagen cercana.
        datos_de_comparacion = {'is_relevant': is_relevant,
                                'dist': dist,
                                'other_filename': other_filename,
                                }
        # Si guardo la distancia puedo ordenar después
        # si es relevante puedo calcular el mAP
        # Y si guardo el nombre del archivo podré dibujar una cajita después.
        distances.append(datos_de_comparacion)
    # Cuando ya vimos las distancias de esta imagen de test a todas las imagenes de training
    # las ordenamos de menor distancia a mayor.
    sorted_distances = sorted(distances, key=lambda d: d['dist'])
    # y seleccionamos las 10 más cercanas.
    distances = sorted_distances[:10]
    return distances, filename


def imprimir_imagenes(distances, filename, i, relevant_counter, str_datadir):
    ## esto imprime las fotitos concatenadas
    ## la primera es la de prueba y las que le siguen son la respuesta a la query
    r = np.random.random()
    ## Sólo para que no dibuje tanto le puse ese límite.
    ## los que tienen relevant counter == 2 son chistosos y no le achuntan a nada
    ## mientras que los que tienen mayor a 10 son aburridos porque le achuntaron a toda la cosa.
    imprimir = False
    if (relevant_counter == 2 and r > 0.7) or relevant_counter > 9:
        imprimir = True
        ## Imprimir sólo algunos que tengan 2 relevantes y todos los que tengan 9 o más relevantes
    if imprimir:
        para_mostrar = [filename]
        for d in distances[:10]:
            para_mostrar.append(os.path.join(str_datadir, 'png_w256', d['other_filename']))
        final_dim = (256 * len(para_mostrar), 256)
        new_im = Image.new('L', final_dim)
        for j, im_file in enumerate(para_mostrar):
            im = Image.open(im_file)
            new_im.paste(im, (j * 256, 0))
        new_im.save('comparado_{}_{}.jpg'.format(relevant_counter, i))
        print('comparado_{}_{}.jpg'.format(relevant_counter, i))


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