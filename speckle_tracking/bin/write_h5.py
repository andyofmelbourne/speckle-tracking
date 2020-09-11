#!/usr/bin/env python
import argparse
from ast import literal_eval
import os
import h5py
import speckle_tracking as st
import numpy as np

def promt_to_create_h5(fnam):
    # check that h5 file exists, if not create it
    outputdir = os.path.split(os.path.abspath(fnam))[0]
    
    if not os.path.exists(fnam):
        yn = input(str(fnam) + ' does not exist. Create it? [y]/n : ')
        if yn.strip() == 'y' or yn.strip() == '' :
            # mkdir if it does not exist
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            
            # make an empty file
            import h5py
            f = h5py.File(fnam, 'w')
            f.close()
        else :
            raise NameError('h5 file does not exist: ' + fnam)

if __name__ == '__main__':
    description = """
    Evaluate a python readable expression and write it to a dataset.

    Example:
    write_h5.py diatom.cxi/speckle_tracking/good_frames range(1,121)
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('dataset', type=str, \
                        help="file name of the *.cxi followed by the dataset location e.g.: foo.cxi/bar/data")
    parser.add_argument('expression', type=str, \
                        help="scalar dataset or python readable expression")
    
    args = parser.parse_args()
    
    # now split the dataset name into filename and dataset location
    # assume the last '.' is before the filename extension
    a = args.dataset.split('.')
    fnam    = '.'.join(a[:-1]) + '.' + a[-1].split('/')[0]
    dataset = args.dataset.split(fnam)[-1]
    
    promt_to_create_h5(fnam)
    
    con = input(
    """WARNING: this uses python eval which could do bad things.
    Are you sure you want to evaluate the following expression? """
    + '\n' + args.expression + '\n y/n?: ')
    
    if con == 'y':
        v = eval(args.expression)
    
    # write to file
    st.write_h5({dataset: v}, fnam=fnam, og='')

