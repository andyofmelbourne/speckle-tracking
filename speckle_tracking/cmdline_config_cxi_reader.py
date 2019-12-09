"""
This is a catch all script to do the following: 

  - get the command line arguments 
  - get the config parameters
  - extract config parameters that look like ``/h5_group/blah`` from the cxi
    file. *But* if there is an roi option then only extract data within the
    roi for datasets that have the same shape as the last two dimensions of
    ``/entry_1/data_1/data`` (a bit hacky). This is dangerous for large 
    datasets...
  - Now convert sample translations into pixel locations. 
  - When the script is finished, write all data into a single h5_group within
    the cxi file and insert roi'd data into the full frames. 

Also we should be careful with scripts that use MPI, we cant have all processes
each load the full dataset of frames, so by default I guess we split over frames.
Maybe we also want the option to split over detector pixels...
"""
import numpy as np
import h5py
import copy
import os

from . import config_reader
from . import cmdline_parser

config_default = {
        'data' : {
            'data'  : '/entry_1/data_1/data',
            'basis' : '/entry_1/instrument_1/detector_1/basis_vectors',
            'z'     : '/entry_1/instrument_1/detector_1/distance',
            'x_pixel_size' : '/entry_1/instrument_1/detector_1/x_pixel_size',
            'y_pixel_size' : '/entry_1/instrument_1/detector_1/y_pixel_size',
            'wav'          : '/entry_1/instrument_1/source_1/wavelength',
            'translations' : '/entry_1/sample_1/geometry/translation'
            },
        'speckle_tracking' : {
            'mask' : '/speckle_tracking/mask',
            'W'    : '/speckle_tracking/whitefield',
            'O'    : '/speckle_tracking/object_map',
            'n0'   : '/speckle_tracking/n0',
            'm0'   : '/speckle_tracking/m0',
            'dxy'  : '/speckle_tracking/object_map_voxel_size',
            'pixel_map' : '/speckle_tracking/pixel_map',
            'xy'   : '/speckle_tracking/pixel_translations',
            'roi'  : '/speckle_tracking/roi'
            }
        }


def get_all(sn, des, exclude=[], config_dirs=None, roi=False):
    """
    exclude can be used to avoid large datasets for example
    """
    # get command line args
    args, params = cmdline_parser.parse_cmdline_args(sn, des, config_dirs=config_dirs)
    
    # get the datasets from the cxi file
    params = config_read_from_h5(params, args.filename, False, True, roi=roi, exclude=exclude)
    
    # now convert from physical to pixel units:
    if 'translation' in params[sn] :
        R_ss_fs, dx, defocus = get_Fresnel_pixel_shifts_cxi(**params[sn])
        params[sn]['defocus']              = defocus
        params[sn]['R_ss_fs']              = R_ss_fs
        params[sn]['magnified_pixel_size'] = dx
    
    # calculate the wavelength if needed
    if 'wavelength' not in params[sn] and 'energy' in params[sn]:
        import scipy.constants as sc
        params[sn]['wavelength'] = sc.h * sc.c / params[sn]['energy']
    return args, params


def write_all(params, filename, output_dict, apply_roi=True):
    # write all datasets to the h5 file
    # but undo all roi stuff
    # and convert pixel shifts to sample translations
    h5_group = params['h5_group']
    
    h5_file = h5py.File(filename, 'r')
    if apply_roi :
        roi      = params['roi']
        shape = h5_file['/entry_1/data_1/data'].shape[-2:]
        N     = h5_file['/entry_1/data_1/data'].shape[0]
        roi_shape = (roi[1]-roi[0], roi[3]-roi[2])
         
        # un-roi all datasets 
        for k in output_dict.keys():
            if type(output_dict[k]) is np.ndarray :
                if (k != 'good_frames') and ('good_frames' in params) and (output_dict[k].shape[0] == len(params['good_frames'])) :
                    print('resizing 0 axis of:', k)
                    temp = np.zeros((N,) + output_dict[k].shape[1:], 
                                    dtype=output_dict[k].dtype)
                    
                    temp[params['good_frames']] = output_dict[k]
                    
                    # now overwrite output array
                    output_dict[k] = temp
                
                if len(output_dict[k].shape) >= 2 :
                    if output_dict[k].shape[-2:] == roi_shape :
                        print('resizing detector axis of:', k)
                        temp = np.zeros(output_dict[k].shape[:-2] + shape, 
                                        dtype=output_dict[k].dtype)

                        temp[..., roi[0]:roi[1], roi[2]:roi[3]] = output_dict[k]
                        
                        # now overwrite output array
                        output_dict[k] = temp
    h5_file.close() 
    
    # un-pixel convert positions
    if 'R_ss_fs' in output_dict :
        output_dict['translation'] = get_Fresnel_pixel_shifts_cxi_inverse(offset_to_zero = True,
                                                                          **params)
    config_reader.write_h5(filename, h5_group, output_dict)
    

def get_Fresnel_pixel_shifts_cxi(
        y_pixel_size=None, x_pixel_size=None, distance=None,
        energy=None, basis_vectors=None, translation=None, 
        defocus=None, good_frames=None, offset_to_zero=True,
        **kwargs):
    import scipy.constants as sc
    du      = np.array([y_pixel_size, x_pixel_size])
    wavelen = sc.h * sc.c / energy
    
    if good_frames is None :
        good_frames = np.arange(len(translation))
    
    if defocus is None :
        defocus = translation[0][2]
    
    b = basis_vectors[good_frames]
    R = translation[good_frames]
    
    # get the magnified sample-shifts 
    # -------------------------------
    # the x and y positions along the pixel directions
    R_ss_fs = np.array([np.dot(b[i], R[i]) for i in range(len(R))])
    R_ss_fs[:, 0] /= du[0]
    R_ss_fs[:, 1] /= du[1]
    
    # I want the x, y coordinates in scaled pixel units
    # divide R by the scaled pixel size
    R_ss_fs /= (defocus / distance) * du
    
    # offset the sample shifts so they start at zero
    if offset_to_zero :
        R_ss_fs[:, 0] -= np.max(R_ss_fs[:, 0])
        R_ss_fs[:, 1] -= np.max(R_ss_fs[:, 1])
    return R_ss_fs, (defocus / distance) * du, defocus

def get_Fresnel_pixel_shifts_cxi_inverse(
        R_ss_fs      = None, 
        y_pixel_size = None, x_pixel_size  = None, distance = None,
        energy       = None, basis_vectors = None, translation       = None, 
        defocus      = None, good_frames   = None, offset_to_zero    = True,
        **kwargs):
    """
    translation must not be truncated by good_frames...
    """
    import scipy.constants as sc
    du      = np.array([y_pixel_size, x_pixel_size])
    wavelen = sc.h * sc.c / energy
    
    if good_frames is None :
        good_frames = np.arange(len(translation))
    
    if defocus is None :
        defocus = translation[0][2]
    
    b = basis_vectors[good_frames]
    
    R_ss_fs_out = R_ss_fs.astype(np.float).copy()
    
    # un-offset
    if offset_to_zero :
        R_ss_fs0, dx, _ = get_Fresnel_pixel_shifts_cxi(
                            y_pixel_size, x_pixel_size , distance, 
                            energy      , basis_vectors, translation,        
                            defocus     , good_frames  , offset_to_zero = False)
        
        #R_ss_fs_out[:, 0] -= np.max(R_ss_fs_out[:, 0])
        #R_ss_fs_out[:, 1] -= np.max(R_ss_fs_out[:, 1])
        
        R_ss_fs_out[:, 0] += np.max(R_ss_fs0[:, 0])
        R_ss_fs_out[:, 1] += np.max(R_ss_fs0[:, 1])
    
    # un-scale
    R_ss_fs_out *= (defocus / distance) * du
    
    # unfortunately we cannot invert from pixel shifts to xyz 
    # this is only possible if the detector lies in the xy plane
    R_ss_fs_out *= du
    
    #print('\ninverting from sample coords to detector coords:')
    for i in range(R_ss_fs_out.shape[0]):
        Ri, r, rank, s = np.linalg.lstsq(b[i][:, :2], R_ss_fs_out[i])
        translation[good_frames[i]][:2] = Ri
        #print(R_ss_fs_out[i], '-->', Ri)
    
    translation[:, 2] = defocus 
    return translation

def roi_extract(f, roi, key, shape):
    fshape = f[key].shape
    # for now any axis within shape will be roi'd
    s = [slice(None) for i in fshape]
    
    if roi is not None and roi is not False and shape is not None :
        for i in range(len(fshape)):
            for j in range(len(shape)):
                if fshape[i] == shape[j]:
                    s[i] = roi[j]
     
    # Hack: this doesn't work for 1d arrays
    # I have no idea why...
    if len(s) == 1:
        return f[key][()]
    return f[key][tuple(s)]


def get_val_h5_new(fnam, val, roi, shape):
    # see if val is one of:
    # 1: python expression
    # 2: hdf path for fnam. e.g. /foo/bar
    # 3: hdf path for another file e.g. loc/foo.cxi/bar
    if type(val) is not str :
        option = 1
        return val
    
    # split val name into filename and dataset location
    # assume the last '.' is before the filename extension
    a       = val.split('.')
    fnam2   = '.'.join(a[:-1]) + '.' + a[-1].split('/')[0]
    dataset = val.split(fnam2)[-1]
    
    if fnam2 != '.' and os.path.exists(fnam2) :
        option = 3
        fnam3  = fnam2
    elif val[0] == '/':
        option = 2
        fnam3  = fnam
        dataset = val
    else :
        # it was just a plain old string
        return val
    
    with h5py.File(fnam3, 'r') as f:
        if val not in f :
            raise KeyError(val + ' not found in file')
        
        return roi_extract(f, roi, val, shape)



def config_read_from_h5(config, h5_file, val_doc_adv=False, 
                        extract=False, roi=False, exclude=[], 
                        flatten=False, update_default={}):
    """
    Same as config_read, but also gets variables from an open h5_file:
        [group-name]
        a = 1.1
        b = /process/blah
        c = fnam.h5/process/blah
    
    Parameters
    ----------
    config : a dictionary of strings / parameters
        filename of the configuration file/s to be parsed, the first sucessfully parsed file is parsed.
    
    h5_file : string or an open hdf5 file 
    
    val_doc_adv : bool
        Toggles the output format of 'config_dict', see below.

    flatten : bool, default (False)
        If True then all key value pairs are returned in the top level 
        of the dictionary.
    
    Returns
    -------
    config_dict : OrderedDict
        If val_doc_adv is True then config_dict is an ordered dictionary of the form:
            output = {group-name: {key : (eval(val), doc_string, is_advanced)}}
        
        Every value in the returned dicionary is a len 3 tuple of the form:
            (val, doc_string, is_advanced)
        
        If the doc string is not suplied in the file then doc_string is None. If 
        val_doc_adv is False then config_dict is an ordered dictionary of the form:
            output = {group-name: {key : eval(val)}}
    
    """
    # if the config is None then load the default
    if config is None :
        config = copy.deepcopy(config_default)
        for sec in update_default.keys() :
            if sec in config_default :
                config[sec].update(update_default[sec])
            else :
                config[sec] = update_default[sec]
    
    # now search for '/' and fetch from the open
    for sec in config.keys():
        
        # get the roi if there
        if roi is not None and roi is not False :
            with h5py.File(h5_file) as f:
                shape = f['/entry_1/data_1/data'].shape
            
            roi = [slice(None) for i in shape]
            
            if val_doc_adv :
                val, doc, adv = config[sec]['roi'][0]
            else :
                val           = config[sec]['roi']
            
            roi1 = get_val_h5_new(h5_file, val, None, None)

            if roi1 is not None :
                roi[1] = slice(roi1[0], roi1[1])
                roi[2] = slice(roi1[2], roi1[3])
             
            if 'good_frames' in config[sec] :
                if val_doc_adv :
                    val, doc, adv = config[sec]['good_frames'][0]
                else :
                    val           = config[sec]['good_frames']
                
                roi0 = get_val_h5_new(h5_file, val, None, None)
                roi[0] = roi0

        else :
            roi, shape = None, None

        for k in config[sec].keys():
            if val_doc_adv :
                val, doc, adv = config[sec][k][0]
            else :
                val = config[sec][k]
            
            if k in exclude :
                continue
            
            valout = get_val_h5_new(h5_file, val, roi, shape)
            
            if val_doc_adv :
                config[sec][k] = (valout, doc, adv)
            else :
                config[sec][k] = valout
    
    if flatten :
        out = {}
        for sec in config.keys():
            for k in config[sec].keys():
                out[k] = config[sec][k]
    else :
        out = config

    return out

