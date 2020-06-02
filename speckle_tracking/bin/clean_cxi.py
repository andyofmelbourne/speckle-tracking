#!/usr/bin/env python
import argparse
import h5py
import tempfile
import shutil
import os


if __name__ == '__main__':
    description = """
    Cleans the hdf5 or cxi file of "stale data" reducing the file size.
    
    This script makes a new temporary file copies over all datasets,
    then renames it to the original file. This is done to
    prevent a feature (bug) of the h5 file format that keeps deleted
    datasets in the file, which leads to large files after a lot of 
    write and delete operations.
    
    Example:
    clean_cxi.py diatom.cxi 
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('cxi_file', type=str, \
                        help="file name of the cxi file")
    
    args = parser.parse_args()
    
    # check if file exists
    if not os.path.exists(args.cxi_file):
        raise NameError('h5 file does not exist: ' + args.cxi_file)
    
    # make a temp file name
    temp_name = next(tempfile._get_candidate_names())
    
    # copy entry_1
    with h5py.File(temp_name, 'w') as temp:
        with h5py.File(args.cxi_file, 'r') as f:
            for key in f.keys():
                f.copy(key, temp['/'])
    
    # now delete orininal file
    shutil.move(temp_name, args.cxi_file)

    print('done')

