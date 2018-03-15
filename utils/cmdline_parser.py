try :
    import ConfigParser as configparser 
except ImportError :
    import configparser

import config_reader
import argparse
import os

# set root
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]

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

def parse_cmdline_args(script_name, description, \
                       create_h5=True, copy_config_to_h5dir = True, \
                       config_dirs = ['process/', 'gui/']):
    """
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('filename', type=str, \
                        help="file name of the *.cxi file")
    parser.add_argument('-c', '--config', type=str, \
                        help="file name of the configuration file")
    
    args = parser.parse_args()
    
    if create_h5 :
        promt_to_create_h5(args.filename)
    
    con_dirs  = [os.path.split(args.filename)[0],] + config_dirs 
    con_fnams = [os.path.join(root, cd+'/'+script_name+'.ini') for cd in con_dirs]
    
    # process config file
    params, fnam = config_reader.config_read(con_fnams)
    
    # copy the config file
    ######################
    if copy_config_to_h5dir and not os.path.exists(con_fnams[0]):
        import shutil
        outputdir = os.path.split(os.path.abspath(args.filename))[0]
        shutil.copy(fnam, outputdir)

    return args, params[script_name]
