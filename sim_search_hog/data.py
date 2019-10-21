'''
Created on Oct 19, 2019

@author: jsaavedr
'''

def read_data(str_file):
    with open(str_file) as f_in:            
        lines = [line.rstrip() for line in f_in]             
    lines_ = [tuple(line.rstrip().split('\t'))  for line in lines ]
    return lines_