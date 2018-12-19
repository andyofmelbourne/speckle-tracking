"""
############################
Match features between frames by maximising their
Pearson correlation coefficient over a small search window. 

Then find the best set of absolute frame positions given their
relative shifts. 

Then build the "atlas" or global view of the object.
############################
we have K (untrusted) features of the form:
    features[k] = n, xi, yi, m, xj, yj

where a feature at pixel (xi, yi) in frame n is near the feature at pixel (xj, yj) in frame m.

---> Update the feature postions within a region:
given
    I_n(x_i)     = I_m(x_j)

minimise (for all k and j) 
    e_k(x_j) = -Pearson(I_n(x_i), I_m(x_j))
    
where 
    Pearson(I(x-x_i), I(x-x_j)) = 
                    [n sum_x w(x) I_n(x-x_i) * I_m(x-x_j)
                     - sum_x w(x) I_n(x-x_i) * sum_x w(x) I_m(x-x_j)] / 
                     [sqrt{n sum_x w(x) I_n(x-x_i)^2 - (sum_x w(x) I_n(x-x_i))^2} 
                    * sqrt{n sum_x w(x) I_m(x-x_j)^2 - (sum_x w(x) I_m(x-x_j))^2}] 

w(x) is a window function and n = sum_x w(x).

---> Find the coordinates:
if   
    I_n(x) = O(x - x_n)
    
then 
    O(x_i - x_n) = O(x_j - x_m)

minimise (for all k)
    e_m(x_l) = sum_k [(x_m - x_n) - (x_j - x_i)]**2 + (sum_l x_l)**2

where k is the feature index and x_l are the unique list of positions.
"""

try :
    import configparser
except ImportError :
    import ConfigParser as configparser

import h5py
import numpy as np
import utils
import os
import time

import pyximport; pyximport.install()
import feature_matching

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Build an atlas from projection images and feature matches')
    parser.add_argument('filename', type=str, \
                        help="file name of the *.pty file")
    parser.add_argument('-c', '--config', type=str, \
                        help="file name of the configuration file")
    
    args = parser.parse_args()
    
    # check that cxi file exists
    if not os.path.exists(args.filename):
        raise NameError('cxi file does not exist: ' + args.filename)
    
    # if config is non then read the default from the *.pty dir
    if args.config is None :
        args.config = os.path.join(os.path.split(args.filename)[0], 'build_atlas.ini')
        if not os.path.exists(args.config):
            args.config = '../process/build_atlas.ini'
    
    # check that args.config exists
    if not os.path.exists(args.config):
        raise NameError('config file does not exist: ' + args.config)
    
    # process config file
    config = configparser.ConfigParser()
    config.read(args.config)
    
    params = utils.parse_parameters(config)
    
    return args, params

def chunkIt(seq, num):
    splits = np.mgrid[0:len(seq):(num+1)*1J].astype(np.int)
    out    = []
    for i in range(splits.shape[0]-1):
        out.append(seq[splits[i]:splits[i+1]])
    return out

def unchunkIt(seq):
    out = []
    for s in seq :
        out += s
    return out

def get_frame_speckle_shifts(frames_to_match, get_frame, h5_features, params):
    # find relative shift between frames k and l given feature list
    h5_features_out = []
    w             = [params['build_atlas']['window'], params['build_atlas']['window']]
    search_window = params['build_atlas']['search_window']
    pairs = []
    # find all features between two frames (minimise frame getting...)
    print('my frames:', rank, frames_to_match )
    for frame1_index in frames_to_match :
        # get the frames
        frame1 = get_frame(frame1_index)
        
        for frame2_index in np.unique(h5_features[:, 3]).astype(np.int):
            # find all features between frame1_index and any other frame
            go_for_it = False
            for fe in h5_features:
                if int(fe[0]) == frame1_index and int(fe[3]) == frame2_index and [int(fe[0]), int(fe[3])] not in pairs :
                    go_for_it = True
                    
                    # remember that we have done this pair
                    pairs.append([int(fe[0]), int(fe[3])])
                    break
            
            if not go_for_it :
                continue
            
            # get the frames
            frame2 = get_frame(frame2_index)
            print('compairing:', frame1_index, frame2_index)
            
            # extract the features to match
            features = []
            ij_grid  = []
            ij2_grid = []
            for fe in h5_features:
                if fe[0] == frame1_index and fe[3] == frame2_index :
                    ij = [int(fe[1]-ROI[0]), int(fe[2]-ROI[2])]
                    ij_grid.append( ij )
                    
                    ij2 = [int(fe[4]-ROI[0]), int(fe[5]-ROI[2])]
                    ij2_grid.append( ij2 )
                    
                    # fftfreq based indexing
                    i_min, i_max = int(max(ij[0]+(1-w[0])//2, 0)), int(min(ij[0]+(w[0]+1)//2, frame1.shape[0]))
                    j_min, j_max = int(max(ij[1]+(1-w[1])//2, 0)), int(min(ij[1]+(w[1]+1)//2, frame1.shape[1]))
                    features.append(frame1[i_min:i_max, j_min:j_max])
            
            # update feature locations within search window
            # calculate the feature correlation maps
            for ij1, ij2, feature in zip(ij_grid, ij2_grid, features) :
                i_min = ij2[0]+(1-search_window)//2
                i_max = ij2[0]+(1+search_window)//2
                j_min = ij2[1]+(1-search_window)//2
                j_max = ij2[1]+(1+search_window)//2
                err = feature_matching.feature_err_map_range(feature, frame2, 
                                                             i_min, i_max, j_min, j_max)
                
                ij2_fit = np.unravel_index(np.argmin(err), err.shape)
                h5_features_out.append([frame1_index, ij1[0]+ROI[0], ij1[1]+ROI[2], 
                                        frame2_index, ij2_fit[0]+ROI[0], ij2_fit[1]+ROI[2]])

                print((ij2[0]+ROI[0], ij2[1]+ROI[2]), '-->', h5_features_out[-1])
    
    return h5_features_out

def calculate_positions(fes):
    """
    e_m(x_l) = sum_k [(x_m - x_n) - (x_j - x_i)]**2 + (sum_l x_l)**2
    """
    # get unique position indexs
    pos = np.unique(np.concatenate((fes[:, 0], fes[:, 3])))
    inds_n = np.zeros((len(fes),), dtype=np.uint)
    inds_m = np.zeros((len(fes),), dtype=np.uint)
    N = len(pos)

    for i, fe in enumerate(fes):
        inds_n[i] = np.where(pos == fe[0])[0][0]
        inds_m[i] = np.where(pos == fe[3])[0][0]
    
    for fe in fes:
        print(fe)
        print(fe[0], fe[3], (fe[4] - fe[1]), (fe[5] - fe[2]))


    def fun(x):
        x, y = x[:N], x[N:]
        xn = x[inds_n]
        xm = x[inds_m]
        yn = y[inds_n]
        ym = y[inds_m]
        err = np.sum( (xm - xn - (fes[:, 4] - fes[:, 1]))**2 \
                    + (ym - yn - (fes[:, 5] - fes[:, 2]))**2)\
             + np.mean(x)**2 + np.mean(y)**2
        return err

    def resid_fes(x):
        x, y = x[:N], x[N:]
        xn = x[inds_n]
        xm = x[inds_m]
        yn = y[inds_n]
        ym = y[inds_m]
        err =         (xm - xn - (fes[:, 4] - fes[:, 1]))**2 \
                    + (ym - yn - (fes[:, 5] - fes[:, 2]))**2
        return err 
    
    import scipy.optimize
    
    x0 = np.zeros((2*pos.shape[0],), dtype=np.float)
    
    d0 = time.time()
    res = scipy.optimize.minimize(fun, x0, options={'disp' : True})
    d1 = time.time()
    print('time to optimise:', d1-d0, 's') 
    
    x, y = res.x[:N], res.x[N:]
    
    print('feature errors:')
    errs_fe = resid_fes(res.x)
    i = np.argsort(errs_fe)[::-1]
    for ii in i:
        print(fes[ii], (round(x[inds_n[ii]]), round(y[inds_n[ii]])), 
                       (round(x[inds_m[ii]]), round(y[inds_m[ii]])), 
                        np.sqrt(errs_fe[ii]/2.))
    
    positions = np.vstack((pos, x, y)).T
    return positions

def calculate_xy_positions(positions, X_fs, z, pix_size, data_len):
    """
    x_n = xpix_n * X / (max(xpix_n) - min(xpix_n)) 
    
    defocus = z * X / [pix_size * (max(xpix_n) - min(xpix_n))]
    """
    trans = np.zeros((data_len, 3), dtype=np.float)
    inds  = positions[:,0].astype(np.int)

    fs = positions[:, 2]
    ss = positions[:, 1]
    
    defocus_x = z * X_fs / (pix_size[1] * (np.max(fs)-np.min(fs)))
    
    trans[inds, 0] = fs * X_fs / (np.max(fs)-np.min(fs))
    trans[inds, 1] = ss * X_fs / (np.max(fs)-np.min(fs))
    trans[:,    2] = defocus_x
    
    basis_vectors = np.zeros((data_len, 2, 3), dtype=np.float)
    basis_vectors[:, 0] = np.array([0, pix_size[0], 0]) 
    basis_vectors[:, 1] = np.array([pix_size[1], 0, 0]) 
    return trans, basis_vectors 

if __name__ == '__main__':
    args, params = parse_cmdline_args()

    # take turns loading data
    for r in range(size):
        if r == rank :
            f = h5py.File(args.filename)
            ################################
            # Get the inputs
            # frames, df, R, O, W, ROI, mask
            ################################
            group = params['build_atlas']['h5_group']
            
            # ROI
            # ------------------
            if params['build_atlas']['roi'] is not None :
                ROI = params['build_atlas']['roi']
            else :
                ROI = [0, f['entry_1/data_1/data'].shape[1], 0, f['entry_1/data_1/data'].shape[2]]
            
            # frames
            # ------------------
            # get the frames to process
            if 'process_3/good_frames' in f :
                good_frames = list(f['process_3/good_frames'][()])
            else :
                good_frames = range(f['entry_1/data_1/data'].shape[0])
            data_len = f['entry_1/data_1/data'].shape[0]
            pix_size = np.array([f['/entry_1/instrument_1/detector_1/y_pixel_size'][()], f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]])
            z  = f['/entry_1/instrument_1/detector_1/distance'][()]
            
            # W
            # ------------------
            # get the whitefield
            W = f[params['build_atlas']['whitefield']][()][ROI[0]:ROI[1], ROI[2]:ROI[3]].astype(np.float)
            W[W == 0] = 1
            
            # mask
            # ------------------
            # mask hot / dead pixels
            if params['build_atlas']['mask'] is None :
                bit_mask = f['entry_1/instrument_1/detector_1/mask'].value
                # hot (4) and dead (8) pixels
                mask     = ~np.bitwise_and(bit_mask, 4 + 8).astype(np.bool) 
            else :
                mask = f[params['build_atlas']['mask']].value
            mask     = mask[ROI[0]:ROI[1], ROI[2]:ROI[3]]

            # features
            # ------------------
            # fr_k, ki, kj, fr_l, li, lj
            h5_features = f[params['build_atlas']['features']][()]

            f.close()
        else :
            comm.barrier()
    
    def get_frame(index):
        frame = None
        for i in range(10):
            try :
                f = h5py.File(args.filename, 'r')
                # apply mask and divide by whitefield
                frame = f['entry_1/data_1/data'][index, ROI[0]:ROI[1], ROI[2]:ROI[3]] 
                frame = frame.astype(np.float64) / W
                frame *= mask
                frame -= ~mask
                f.close()
            except OSError :
                time.sleep(0.1)
            
            if frame is not None :
                return frame
        raise OSError('could not open h5 file...')
    
    if rank == 0 :
        frames_to_match = chunkIt(np.unique(h5_features[:, 0]).astype(np.int), size)
    else :
        frames_to_match = None

    frames_to_match = comm.scatter(frames_to_match, root = 0)
    
    h5_features_out = get_frame_speckle_shifts(frames_to_match, get_frame, h5_features, params)
    
    h5_features_out = comm.gather(h5_features_out, root = 0)
    if rank == 0 :
        h5_features_out = np.array(unchunkIt(h5_features_out))
        
        positions = calculate_positions(h5_features_out)
        
        translations, basis_vectors = calculate_xy_positions(positions, \
                                          params['build_atlas']['fs_range'], z, pix_size, data_len)
        
        # build the atlas
        inds   = positions[:, 0].astype(np.int)
        frames = np.array([get_frame(i) for i in inds])
        atlas  = feature_matching.build_atlas(frames, positions[:, 1:])
        
        # output
        ########
        # write new feature locations
        # write the atlas
        
        f = h5py.File(args.filename)
        
        g = f
        outputdir = os.path.split(args.filename)[0]
        
        group = params['build_atlas']['h5_group']
        if group not in g:
            print(g.keys())
            g.create_group(group)

        key = params['build_atlas']['h5_group']+'/translations'
        if key in g :
            del g[key]
        g[key] = translations
        
        key = params['build_atlas']['h5_group']+'/basis_vectors'
        if key in g :
            del g[key]
        g[key] = basis_vectors
        
        key = params['build_atlas']['h5_group']+'/pix_positions'
        if key in g :
            del g[key]
        g[key] = positions
        
        key = params['build_atlas']['h5_group']+'/features'
        if key in g :
            del g[key]
        g[key] = h5_features_out
        
        key = params['build_atlas']['h5_group']+'/atlas'
        if key in g :
            del g[key]
        g[key] = atlas
        
        g.close()
        
        # copy the config file
        ######################
        try :
            import shutil
            shutil.copy(args.config, outputdir)
        except Exception as e :
            print(e)
