#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 14:39:56 2018
@author: jose.saavedra

This implements basic operations on a cnn

"""
from . import configuration as conf
from . import cnn_model as model
from . import data as data
from . import imgproc
import tensorflow as tf
import numpy as np
import os

         
class CNN:
    def __init__(self, str_config, params):        
        self.initFromCkpt = False         
        self.ckpt_file = None
        if 'ckpt' in params:
            self.initFromCkpt = True
            self.ckpt_file = params['ckpt']
        #reading configuration file    
        self.configuration = conf.ConfigurationFile(str_config, params['modelname'])
        self.modelname = self.configuration.get_model_name()
        self.device = params['device']
        self.processFun = imgproc.get_process_fun(self.configuration.get_process_fun())
        #validatitn snapShotDir
        assert os.path.exists(self.configuration.get_snapshot_dir()), "Path {} does not exist".format(self.configuration.get_snapshot_dir())
        #if not os.path.exists(os.path.dirname(self.configuration.getSnapshotDir())) :
        #    os.makedirs(os.path.dirname(self.configuration.getSnapshotDir()))        
        #metadata
        filename_mean = os.path.join(self.configuration.get_data_dir(), "mean.dat")
        metadata_file = os.path.join(self.configuration.get_data_dir(), "metadata.dat")
        #reading metadata
        self.image_shape = np.fromfile(metadata_file, dtype=np.int32)
        #         
        print("image shape: {}".format(self.image_shape))
        #load mean
        mean_img = np.fromfile(filename_mean, dtype=np.float32)
        self.mean_img = np.reshape(mean_img, self.image_shape.tolist())    
        #defining files for training and test
        self.filename_train = os.path.join(self.configuration.get_data_dir(), "train.tfrecords")
        self.filename_test = os.path.join(self.configuration.get_data_dir(), "test.tfrecords")         
        #print(" mean {}".format(self.mean_img.shape))        
                    
    def train(self):
        """training"""
        #-using device gpu or cpu
        with tf.device(self.device):
            estimator_config = tf.estimator.RunConfig( model_dir = self.configuration.get_snapshot_dir(),
                                                       save_checkpoints_steps=self.configuration.get_snapshot_steps(),
                                                       keep_checkpoint_max=10)            
            classifier = tf.estimator.Estimator( model_fn = model.model_fn,
                                                 config = estimator_config,
                                                 params = {'learning_rate' : self.configuration.get_learning_rate(),
                                                           'number_of_classes' : self.configuration.get_number_of_classes(),
                                                           'image_shape' : self.image_shape,
                                                           'number_of_channels' : self.configuration.get_number_of_channels(),
                                                           'model_dir': self.configuration.get_snapshot_dir(),
                                                           'ckpt' : self.ckpt_file,
                                                           'arch' : self.configuration.get_architecture()
                                                           }
                                                 )
            #
            tf.logging.set_verbosity(tf.logging.INFO) # Just to have some logs to display for demonstration
            #training
            train_spec = tf.estimator.TrainSpec(input_fn = lambda: data.input_fn(self.filename_train, 
                                                                             self.image_shape, 
                                                                             self.mean_img, 
                                                                             is_training = True, 
                                                                             configuration =  self.configuration),
                                                 max_steps = self.configuration.get_number_of_iterations())
            #max_steps is not useful when inherited checkpoint is used
            eval_spec = tf.estimator.EvalSpec(input_fn = lambda: data.input_fn(self.filename_test, 
                                                                           self.image_shape, 
                                                                           self.mean_img, 
                                                                           is_training = False, 
                                                                           configuration =  self.configuration),                                             
                                              throttle_secs = self.configuration.get_test_time())
            #
            tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
        
    def test(self):
        """test checkpoint exist """
        assert os.path.exists(os.path.join(self.configuration.get_snapshot_dir(), "checkpoint")), "Checkpoint file does not exist in {}".format(self.configuration.get_snapshot_dir())
        """testing"""
        with tf.device(self.device):
            estimator_config = tf.estimator.RunConfig(model_dir = self.configuration.get_snapshot_dir())
            classifier = tf.estimator.Estimator(model_fn = model.model_fn,
                                                 config = estimator_config,
                                                 params = {'learning_rate' : self.configuration.get_learning_rate(),
                                                           'number_of_classes' : self.configuration.get_number_of_classes(),
                                                           'image_shape' : self.image_shape,
                                                           'number_of_channels' : self.configuration.get_number_of_channels(),
                                                           'model_dir': self.configuration.get_snapshot_dir(),
                                                           'ckpt' : self.ckpt_file,
                                                           'arch' : self.configuration.get_architecture()
                                                           }
                                                 ) 
            result = classifier.evaluate(input_fn=lambda: data.input_fn(self.filename_test,
                                                                   self.image_shape,
                                                                   self.mean_img,
                                                                   is_training = False,
                                                                   configuration =  self.configuration),
                                                                   checkpoint_path = self.ckpt_file)
            print(result)
           
    def predict(self, filename):
        """
        This function is deprectated, prefer to use fast_predictor
        """        
        """test checkpoint exist """        
        assert os.path.exists(os.path.join(self.configuration.get_snapshot_dir(), "checkpoint")), "Checkpoint file does not exist in {}".format(self.configuration.get_snapshot_dir())
        classifier = tf.estimator.Estimator(model_fn = model.model_fn,
                                             model_dir = self.configuration.get_snapshot_dir(),
                                             params = {'learning_rate' : self.configuration.get_learning_rate(),
                                                       'number_of_classes' : self.configuration.get_number_of_classes(),
                                                       'image_shape' : self.image_shape,
                                                       'number_of_channels' : self.configuration.get_number_of_channels(),                                                       
                                                       'arch' : self.configuration.get_architecture()
                                                       })        
        #
        tf.logging.set_verbosity(tf.logging.INFO) # Just to have some logs to display for demonstration
                 
        input_image =  data.input_fn_for_prediction(filename,
                                               self.image_shape,
                                               self.mean_img,
                                               self.configuration.get_number_of_channels(),
                                               self.processFun)
        
           
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x=input_image,
                num_epochs=1,
                shuffle=False
                )                
        #classifier could use checkpoint_path to define the checkpoint to be used
        predicted_result = list(classifier.predict(input_fn = predict_input_fn))
        return predicted_result
        
    def predict_on_list(self, list_of_images) :
        """test checkpoint exist """
        assert os.path.exists(os.path.join(self.configuration.get_snapshot_dir(), "checkpoint")), "Checkpoint file does not exist in {}".format(self.configuration.get_snapshot_dir())
        classifier = tf.estimator.Estimator(model_fn = model.model_fn,
                                             model_dir = self.configuration.get_snapshot_dir(),
                                             params = {'learning_rate' : self.configuration.get_learning_rate(),
                                                       'number_of_classes' : self.configuration.get_number_of_classes(),
                                                       'image_shape' : self.image_shape,
                                                       'number_of_channels' : self.configuration.get_number_of_classes(),
                                                       'arch' : self.configuration.get_architecture()
                                                       })
        #
        tf.logging.set_verbosity(tf.logging.INFO) # Just to have some logs to display for demonstration
        batch_of_images =  data.input_fn_for_prediction_on_list(list_of_images, 
                                                            self.image_shape, 
                                                            self.mean_img, 
                                                            self.configuration.get_number_of_channels(),
                                                            self.processFun)
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x=batch_of_images,
                num_epochs=1,
                shuffle=False)
        #classifier could use checkpoint_path to define the checkpoint to be used
        predicted_result = list(classifier.predict(input_fn = predict_input_fn, yield_single_examples = False))
        return predicted_result
        
    def save_model(self):
        """save model for prediction """
        assert os.path.exists(os.path.join(self.configuration.get_snapshot_dir(), "checkpoint")), "Checkpoint file does not exist in {}".format(self.configuration.get_snapshot_dir())
        classifier = tf.estimator.Estimator(model_fn = model.model_fn,
                                             model_dir = self.configuration.get_snapshot_dir(),
                                             params = {'learning_rate' : self.configuration.get_learning_rate(),
                                                       'number_of_classes' : self.configuration.get_number_of_classes(),
                                                       'image_shape' : self.image_shape,
                                                       'number_of_channels' : self.configuration.get_number_of_channels(),                                                       
                                                       'arch' : self.configuration.get_architecture()
                                                       })        
        #
        def serving_input_receiver_fn() :                            
            feat_spec = tf.placeholder(dtype=tf.float32, shape=[None, self.image_shape[0], self.image_shape[1], self.image_shape[2]])            
            return tf.estimator.export.TensorServingInputReceiver(feat_spec, feat_spec)
           
        str_model = classifier.export_saved_model(self.configuration.get_data_dir(), serving_input_receiver_fn)
        final_str_model = os.path.join(self.configuration.get_data_dir(), "cnn-model")
        os.rename(str_model, final_str_model)
        print("The models was successfully saved at {}".format(final_str_model))
        