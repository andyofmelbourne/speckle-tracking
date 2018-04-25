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

import config_reader
import cmdline_parser


def get_all(sn, des, exclude=[]):
    """
    exclude can be used to avoid large datasets for example
    """
    # get command line args
    args, params = cmdline_parser.parse_cmdline_args(sn, des)
    
    # get the datasets from the cxi file
    params = config_read_from_h5(params, args.filename, False, True, True, exclude=exclude)
    
    # now convert from physical to pixel units:
    R_ss_fs, dx = get_Fresnel_pixel_shifts_cxi(**params[sn])
    params[sn]['R_ss_fs']              = R_ss_fs
    params[sn]['magnified_pixel_size'] = dx
    return args, params


def write_all(params, filename, output_dict, apply_roi=True):
    # write all datasets to the h5 file
    # but undo all roi stuff
    # and convert pixel shifts to sample translations
    h5_group = params['h5_group']
    roi      = params['roi']
    
    import h5py
    h5_file = h5py.File(filename, 'r')
    if apply_roi :
        shape = h5_file['/entry_1/data_1/data'].shape[-2:]
        roi_shape = (roi[1]-roi[0], roi[3]-roi[2])
         
        # un-roi all datasets 
        for k in output_dict.keys():
            if type(output_dict[k]) is np.ndarray :
                if len(output_dict[k].shape) >= 2 :
                    if output_dict[k].shape[-2:] == roi_shape :
                        try :
                            temp = h5_file[h5_group+'/'+k][()]
                        except :
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
        good_frames = len(translation)

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
    return R_ss_fs, (defocus / distance) * du

def get_Fresnel_pixel_shifts_cxi_inverse(
        R_ss_fs      = None, 
        y_pixel_size = None, x_pixel_size  = None, distance = None,
        energy       = None, basis_vectors = None, translation       = None, 
        defocus      = None, good_frames   = None, offset_to_zero    = True,
        **kwargs):
    import scipy.constants as sc
    du      = np.array([y_pixel_size, x_pixel_size])
    wavelen = sc.h * sc.c / energy
    
    if good_frames is None :
        good_frames = len(translation)
    
    if defocus is None :
        defocus = translation[0][2]
    
    b = basis_vectors[good_frames]
    R = translation[good_frames]
    
    R_ss_fs_out = R_ss_fs.astype(np.float).copy()
    
    # un-offset
    if offset_to_zero :
        R_ss_fs0, dx = get_Fresnel_pixel_shifts_cxi(
                            y_pixel_size, x_pixel_size , distance, 
                            energy      , basis_vectors, translation,        
                            defocus     , good_frames  , offset_to_zero = False)
        
        R_ss_fs_out[:, 0] -= np.max(R_ss_fs_out[:, 0])
        R_ss_fs_out[:, 1] -= np.max(R_ss_fs_out[:, 1])
        
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
        R[good_frames[i]][:2] = Ri
        #print(R_ss_fs_out[i], '-->', Ri)
     
    return R

def get_val_h5(h5, val, r, shape, extract, k):
    if val not in h5 :
        raise KeyError(val + ' not found in file')
    
    if r is not None :
        if extract :
            if (len(h5[val].shape) >= 2) and (h5[val].shape[-2:] == shape):
                valout = h5[val][..., r[0]:r[1], r[2]:r[3]]
            else :
                valout = h5[val][()]
        else :
            valout = h5[val]
    else :
        if h5[val].size < 1e5 or extract :
            valout = h5[val][()]
        else :
            valout = h5[val]

    # special case for the bit mask
    if (val == 'entry_1/instrument_1/detector_1/mask'  or \
        val == 'entry_1/instrument_1/detector_1/mask') and \
        type(valout) is np.ndarray :
            # hot (4) and dead (8) pixels
            valout   = ~np.bitwise_and(valout, 4 + 8).astype(np.bool) 
    return valout

def config_read_from_h5(config, h5_file, val_doc_adv=False, extract=False, roi=False, exclude=[]):
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
    import h5py
    
    # open the h5 file if h5_file is a string
    close   = False
    if type(h5_file) is str :
        h5_file = h5py.File(h5_file, 'r')
        close   = True
    
    # get the roi if there
    r = None
    if roi :
        for group in config.keys():
            if 'roi' in config[group] :
                r = config[group]['roi']
                break
            elif 'ROI' in config[group] :
                r = config[group]['ROI']
                break
        # now get the shape of datasets that 
        # this will apply to:
        shape = h5_file['/entry_1/data_1/data'].shape[-2:]
    
    # now search for '/' and fetch from the open
    for sec in config.keys():
        for k in config[sec].keys():
            if val_doc_adv :
                val, doc, adv = config[sec][k][0]
            else :
                val = config[sec][k]

            if k in exclude :
                continue
            
            # extract from same file
            if type(val) is str and val[0] == '/': 
                valout = get_val_h5(h5_file, val, r, shape, extract, k)
            
            # extract from different file *.h5
            elif type(val) is str and '.h5/' in val:
                fn, path = val.split('.h5/')
                f = h5py.File(fn+'.h5', 'r') 
                valout = get_val_h5(f, path, r, shape, extract, k)
                f.close()
            
            # extract from different file *.cxi
            elif type(val) is str and '.cxi/' in val:
                fn, path = val.split('.cxi/')
                f = h5py.File(fn+'.cxi', 'r') 
                valout = get_val_h5(f, path, r, shape, extract, k)
                f.close()
            else :
                valout = None
            
            if valout is None :
                continue 
            
            if val_doc_adv :
                config[sec][k] = (valout, doc, adv)
            else :
                config[sec][k] = valout
    
    if close :
        h5_file.close()
    return config

