#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:00:28 2018

@author: jsaavedr
"""
from configparser import SafeConfigParser

class ConfigurationFile:
    """
     An instance of ConfigurationFile contains required parameters to train a 
     convolutional neural network
    """    
    def __init__(self, str_config, modelname):
        config = SafeConfigParser()
        config.read(str_config)
        self.sections = config.sections()                
        if modelname in self.sections:
            try :
                self.modelname = modelname                
                self.arch = config.get(modelname, "ARCH")
                self.process_fun = 'default'
                if 'PROCESS_FUN' in config[modelname] is not None :
                    self.process_fun = config[modelname]['PROCESS_FUN']                                                    
                self.number_of_classes = config.getint(modelname,"NUM_CLASSES")
                self.number_of_iterations= config.getint(modelname,"NUM_ITERATIONS")
                #self.dataset_size = config.getint(modelname, "DATASET_SIZE")
                #self.test_size = config.getint(modelname, "TEST_SIZE")
                self.batch_size = config.getint(modelname, "BATCH_SIZE")
                #self.estimated_number_of_batches =  int ( float(self.dataset_size) / float(self.batch_size) )
                #self.estimated_number_of_batches_test = int ( float(self.test_size) / float(self.batch_size) )
                #snapshot time sets when temporal weights are saved (in steps)
                self.snapshot_steps = config.getint(modelname, "SNAPSHOT_STEPS")
                #test time sets when test is run (in seconds)
                self.test_time = config.getint(modelname, "TEST_TIME")
                self.lr = config.getfloat(modelname, "LEARNING_RATE")
                #snapshot folder, where training data will be saved
                self.snapshot_prefix = config.get(modelname, "SNAPSHOT_DIR")
                #number of estimated epochs
                #self.number_of_epochs = int ( float(self.number_of_iterations) / float(self.estimated_number_of_batches) )
                #folder where tf data is saved. Used for training and testing
                self.data_dir = config.get(modelname,"DATA_DIR")
                self.channels = config.getint(modelname,"CHANNELS")                
                
                assert(self.channels == 1 or self.channels == 3)                
            except Exception:
                raise ValueError("something wrong with configuration file " + str_config)
        else:
            raise ValueError(" {} is not a valid section".format(modelname))
        
    def get_model_name(self):
        return self.modelname
    
    def get_architecture(self) :
        return self.arch
    
    def get_process_fun(self):
        return self.process_fun
       
    def get_number_of_classes(self) :
        return self.number_of_classes
    
    def get_number_of_iterations(self):
        return self.number_of_iterations
    
#     def get_number_of_epochs(self):
#         return self.number_of_epochs
    
#     def get_data_size(self):
#         return self.dataset_size
    
    def get_batch_size(self):
        return self.batch_size
    
#     def get_number_of_batches(self):
#         return self.estimated_number_of_batches
    
#     def get_number_of_batches_for_test(self):
#         return self.estimated_number_of_batches_test
    
    def get_snapshot_steps(self):
        return self.snapshot_steps
    
    def get_test_time(self):
        return self.test_time
    
    def get_snapshot_dir(self):
        return self.snapshot_prefix
    
    def get_number_of_channels(self):
        return self.channels
    
    def get_data_dir(self):
        return self.data_dir
    
    def get_learning_rate(self):
        return self.lr    
    
    def is_a_valid_section(self, str_section):
        return str_section in self.sections
    
    def show(self):
        print("ARCH: {}".format(self.get_architecture()))
        print("NUM_ITERATIONS: {}".format(self.get_number_of_iterations()))        
        print("LEARNING_RATE: {}".format(self.get_learning_rate()))                
        print("SNAPSHOT_DIR: {}".format(self.get_snapshot_dir()))
        print("DATA_DIR: {}".format(self.get_data_dir()))