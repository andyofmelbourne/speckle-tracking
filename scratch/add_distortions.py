"""
############################
Calculate the pixel distortions, parameterised by 2D polynomial coefficients, 
and the frame positions given a set of matched features between pairs of frames.

Additionally: output the merged "atlas" or global view of the object.
############################

We have K sets of matched features:
    features[k] = n, xi, yi, m, xj, yj

where a feature at pixel (xi, yi) in frame n has been found at pixel (xj, yj) in frame m.
our model for the intensities seen in each frame are
    I_n(x) = O(x - x_n + u(x))

therefore:
    I_n(x_i) = I_m(x_j)
    O(x_i - x_n + u(x_i)) = O(x_j - x_m + u(x_j))

therefore:
    x_m - x_n + x_i - x_j + ux(x_i, y_i) - ux(x_j, y_j) = 0 
    y_m - y_n + y_i - y_j + uy(x_i, y_i) - uy(x_j, y_j) = 0 

we aim to solve these equations, but keep ux and uy under control. 
I propose the following cost function:
    
    e(x, y, a, b) = \sum_k [
                    (x_m - x_n + x_i - x_j + ux(x_i, y_i) - ux(x_j, y_j))**2
                  + (y_m - y_n + y_i - y_j + uy(x_i, y_i) - ux(x_j, y_j))**2 ]
                  + \iint (ux(x, y))^2 dx dy        # keep ux from going wild
                  + \iint (uy(x, y))^2 dx dy        # keep uy from going wild

where x_m, x_n, x_i, x_j are all functions of the feature index k,

and
    ux(x_i, y_i) = sum_kk' a_kk' x^k y^k'
    uy(x_i, y_i) = sum_kk' b_kk' x^k y^k'
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

from build_atlas import calculate_xy_positions

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


def calculate_positions_distortions(pos, fes, roi, K):
    import scipy.optimize
    import pyximport; pyximport.install()
    import poly_utils
    from numpy.polynomial.polynomial import polyval2d

    str_sm = 1.0

    N  = np.max(pos[:, 0])
    NN = len(pos)

    # here we use [ss, fs] --> [x, y] ordering
    xmin, xmax = (1-roi[1]+roi[0])//2, (1+roi[1]-roi[0])//2
    ymin, ymax = (1-roi[3]+roi[2])//2, (1+roi[3]-roi[2])//2
    
    # we need the pos --> index mapping
    pos_inv = np.zeros(int(pos[:, 0].max()+1), dtype=np.uint16)
    for i, p in enumerate(pos[:, 0]):
        pos_inv[int(p)] = i
    
    a  = np.zeros((K, K), dtype=np.float)
    ax = np.zeros((K, K), dtype=np.float)
    ay = np.zeros((K, K), dtype=np.float)
    
    mask = np.ones_like(a).astype(np.bool)
    mask[0, 0] = False # no phase offset
    mask[1, 0] = False # no x-shift
    mask[0, 1] = False # no y-shift
    mask[2, 0] = False # no x-scale
    mask[0, 2] = False # no y-scale
    
    ct = np.zeros((2*K-1, 2*K-1), dtype=np.float)
    bt = np.zeros((2*K-1,), dtype=np.float)
    
    # regularise by distance from the centre
    dist = ((fes[:, 1]+xmin)/float(xmax))**2 + ((fes[:, 4]+xmin)/float(xmax))**2 \
          +((fes[:, 2]+ymin)/float(ymax))**2 + ((fes[:, 5]+ymin)/float(ymax))**2
    reg  = np.exp(-dist**2 / (8. * 1.0**2))
    
    def fun2_xy(xx, ct, bt, ax, ay, a, dist):
        x_n, y_n, a_k = xx[:NN], xx[NN:2*NN], xx[2*NN:]
        
        a[mask] = xx[2*NN:]
        
        ax = poly_utils.polyder2d(a, ax, 0)
        ay = poly_utils.polyder2d(a, ay, 1)
        
        # cost: 
        # (features[k] = n, xi, yi, m, xj, yj)
        # x_m - x_n + x_i - x_j + ux(x_i, y_i) - ux(x_j, y_j) = 0 
        # y_m - y_n + y_i - y_j + uy(x_i, y_i) - uy(x_j, y_j) = 0 
        
        temp1 = x_n[pos_inv[fes[:, 3].astype(np.int)]] - x_n[pos_inv[fes[:, 0].astype(np.int)]]  \
             + fes[:, 1] - fes[:, 4] \
             + polyval2d((fes[:, 1]+xmin)/float(xmax), (fes[:, 2]+ymin)/float(ymax), ax) \
             - polyval2d((fes[:, 4]+xmin)/float(xmax), (fes[:, 5]+ymin)/float(ymax), ax) 
        
        temp2 = y_n[pos_inv[fes[:, 3].astype(np.int)]] - y_n[pos_inv[fes[:, 0].astype(np.int)]]  \
             + fes[:, 2] - fes[:, 5] \
             + polyval2d((fes[:, 1]+xmin)/float(xmax), (fes[:, 2]+ymin)/float(ymax), ay) \
             - polyval2d((fes[:, 4]+xmin)/float(xmax), (fes[:, 5]+ymin)/float(ymax), ay) 
        
        cost = np.sum( (temp1**2 + temp2**2) * reg )
        
        # integrate:
        ct    = poly_utils.polymul2d(a, a, ct)
        cost += str_sm *poly_utils.polyint2d(ct, bt, -1., 1., -1., 1.)
        return cost
        
    def err_poly(xx, ct, bt, ax, ay, a):
        x_n, y_n, a_k = xx[:NN], xx[NN:2*NN], xx[2*NN:]
        
        a[mask] = xx[2*NN:]
        
        ax = poly_utils.polyder2d(a, ax, 0)
        ay = poly_utils.polyder2d(a, ay, 1)
        
        ct    = poly_utils.polymul2d(a, a, ct)
        cost  = str_sm *poly_utils.polyint2d(ct, bt, -1., 1., -1., 1.)
        return cost

    def err_xy(xx, ct, bt, ax, ay, a):
        x_n, y_n, a_k = xx[:NN], xx[NN:2*NN], xx[2*NN:]
        
        a[mask] = xx[2*NN:]
        
        ax = poly_utils.polyder2d(a, ax, 0)
        ay = poly_utils.polyder2d(a, ay, 1)
        
        # cost: 
        # (features[k] = n, xi, yi, m, xj, yj)
        # x_m - x_n + x_i - x_j + ux(x_i, y_i) - ux(x_j, y_j) = 0 
        # y_m - y_n + y_i - y_j + uy(x_i, y_i) - uy(x_j, y_j) = 0 
        
        temp1 = x_n[pos_inv[fes[:, 3].astype(np.int)]] - x_n[pos_inv[fes[:, 0].astype(np.int)]]  \
             + fes[:, 1] - fes[:, 4] \
             + polyval2d((fes[:, 1]+xmin)/float(xmax), (fes[:, 2]+ymin)/float(ymax), ax) \
             - polyval2d((fes[:, 4]+xmin)/float(xmax), (fes[:, 5]+ymin)/float(ymax), ax) 
        
        temp2 = y_n[pos_inv[fes[:, 3].astype(np.int)]] - y_n[pos_inv[fes[:, 0].astype(np.int)]]  \
             + fes[:, 2] - fes[:, 5] \
             + polyval2d((fes[:, 1]+xmin)/float(xmax), (fes[:, 2]+ymin)/float(ymax), ay) \
             - polyval2d((fes[:, 4]+xmin)/float(xmax), (fes[:, 5]+ymin)/float(ymax), ay) 
        
        return temp1**2 + temp2**2
        

    import time
    x0 = np.concatenate((pos[:,1], pos[:, 2], np.zeros(K**2 - 5, dtype=np.float)))

    d0 = time.time()
    res = scipy.optimize.minimize(fun2_xy, x0, args=(ct, bt, ax, ay, a, dist), options={'disp' : True})
    d1 = time.time()
    print('time to optimise:', d1-d0, 's') 

    # print the features with the highest error:
    err = err_xy(res.x, ct, bt, ax, ay, a)
    err_poly = err_poly(res.x, ct, bt, ax, ay, a)
    print('features, highest error --> lowest error:')
    print('total position error:', np.sum(err))
    print('total polyint  error:', err_poly)
    i = np.argsort(err)[::-1]
    for ii in i :
        print(fes[ii], np.sqrt(err[ii]/2.))
    
    xx = res.x
    x_n, y_n = xx[:NN], xx[NN:2*NN]
    
    a[mask] = xx[2*NN:]
    ax = poly_utils.polyder2d(a, ax, 0)
    ay = poly_utils.polyder2d(a, ay, 1)

    print(res)
    print('\nss poly coeffs:')
    print(ax)
    
    print('\nfs poly coeffs:')
    print(ay)

    pos_out = pos.copy()
    pos_out[:,1] = x_n
    pos_out[:,2] = y_n
    
    i, j = np.ogrid[-1:1:(xmax-xmin)*1j, -1:1:(ymax-ymin)*1j]
    uss = polyval2d(i, j, ax) 
    ufs = polyval2d(i, j, ay)
    
    # generate the new feature positions
    # I_m(xj) = I_n(xi)
    # O(xj - x_m + u(xj)) = O(xi - x_n + u(xi))
    # xj + ux(x_j, y_j) = x_m - x_n + x_i + ux(x_i, y_i)
    # so we need the inverse of x - ux and y - uy 
    #fes_out = np.empty_like(fes)
    #for k, fe in enumerate(fes):
    #    xi, yi = fe[1:3]
    #    xj, yj = fe[4:6]
    #    uss_t  = i - uss
    #    ufs_t  = j - ufs
        
    #    ind = np.argmin(np.abs(uss_t - x


    return uss, ufs, pos_out

def calculate_positions_distortions_old(pos, fes, roi, K):
    import scipy.optimize
    import pyximport; pyximport.install()
    import poly_utils
    from numpy.polynomial.polynomial import polyval2d

    str_sm = 0.1

    N  = np.max(pos[:, 0])
    NN = len(pos)

    # here we use [ss, fs] --> [x, y] ordering
    xmin, xmax = (1-roi[1]+roi[0])//2, (1+roi[1]-roi[0])//2
    ymin, ymax = (1-roi[3]+roi[2])//2, (1+roi[3]-roi[2])//2
    
    # we need the pos --> index mapping
    pos_inv = np.zeros(int(pos[:, 0].max()+1), dtype=np.uint16)
    for i, p in enumerate(pos[:, 0]):
        pos_inv[int(p)] = i
    
    ct = np.zeros((2*K-1, 2*K-1), dtype=np.float)
    bt = np.zeros((2*K-1,), dtype=np.float)
    
    def fun2_xy(xx, ct, bt):
        x_n, y_n, a_k, b_k = xx[:NN], xx[NN:2*NN], xx[2*NN:2*NN + K**2 -2], xx[2*NN + K**2 - 2:]
        
        a_k      = np.concatenate(([0,], a_k[:K-1], [0,], a_k[K-1:])).reshape((K, K))
        b_k      = np.concatenate(([0,0], b_k)).reshape((K, K))
        
        # cost: 
        # (features[k] = n, xi, yi, m, xj, yj)
        # x_m - x_n + x_i - x_j + ux(x_i, y_i) - ux(x_j, y_j) = 0 
        # y_m - y_n + y_i - y_j + uy(x_i, y_i) - uy(x_j, y_j) = 0 
        
        temp1 = x_n[pos_inv[fes[:, 3].astype(np.int)]] - x_n[pos_inv[fes[:, 0].astype(np.int)]]  \
             + fes[:, 1] - fes[:, 4] \
             + polyval2d((fes[:, 1]+xmin)/float(xmax-xmin), (fes[:, 2]+ymin)/float(ymax-ymin), a_k) \
             - polyval2d((fes[:, 4]+xmin)/float(xmax-xmin), (fes[:, 5]+ymin)/float(ymax-ymin), a_k) 
        
        temp2 = y_n[pos_inv[fes[:, 3].astype(np.int)]] - y_n[pos_inv[fes[:, 0].astype(np.int)]]  \
             + fes[:, 2] - fes[:, 5] \
             + polyval2d((fes[:, 1]+xmin)/float(xmax-xmin), (fes[:, 2]+ymin)/float(ymax-ymin), b_k) \
             - polyval2d((fes[:, 4]+xmin)/float(xmax-xmin), (fes[:, 5]+ymin)/float(ymax-ymin), b_k) 
        
        cost = np.sum(temp1**2 + temp2**2)
        
        # integrate:
        ct    = poly_utils.polymul2d(a_k, a_k, ct)
        cost += str_sm *poly_utils.polyint2d(ct, bt, -1., 1., -1., 1.)
        ct    = poly_utils.polymul2d(b_k, b_k, ct)
        cost += str_sm *poly_utils.polyint2d(ct, bt, -1., 1., -1., 1.)
        return cost

    def err_xy(xx, ct, bt):
        x_n, y_n, a_k, b_k = xx[:NN], xx[NN:2*NN], xx[2*NN:2*NN + K**2 -2], xx[2*NN + K**2 - 2:]
        
        a_k      = np.concatenate(([0,], a_k[:K-1], [0,], a_k[K-1:])).reshape((K, K))
        b_k      = np.concatenate(([0,0], b_k)).reshape((K, K))
        
        # cost: 
        # (features[k] = n, xi, yi, m, xj, yj)
        # x_m - x_n + x_i - x_j + ux(x_i, y_i) - ux(x_j, y_j) = 0 
        # y_m - y_n + y_i - y_j + uy(x_i, y_i) - uy(x_j, y_j) = 0 
        
        temp1 = x_n[pos_inv[fes[:, 3].astype(np.int)]] - x_n[pos_inv[fes[:, 0].astype(np.int)]]  \
             + fes[:, 1] - fes[:, 4] \
             + polyval2d((fes[:, 1]+xmin)/float(xmax-xmin), (fes[:, 2]+ymin)/float(ymax-ymin), a_k) \
             - polyval2d((fes[:, 4]+xmin)/float(xmax-xmin), (fes[:, 5]+ymin)/float(ymax-ymin), a_k) 
        
        temp2 = y_n[pos_inv[fes[:, 3].astype(np.int)]] - y_n[pos_inv[fes[:, 0].astype(np.int)]]  \
             + fes[:, 2] - fes[:, 5] \
             + polyval2d((fes[:, 1]+xmin)/float(xmax-xmin), (fes[:, 2]+ymin)/float(ymax-ymin), b_k) \
             - polyval2d((fes[:, 4]+xmin)/float(xmax-xmin), (fes[:, 5]+ymin)/float(ymax-ymin), b_k) 
        
        return temp1**2 + temp2**2
        

    import time
    x0 = np.concatenate((pos[:,1], pos[:, 2], np.zeros(2 * K**2 - 4, dtype=np.float)))

    d0 = time.time()
    res = scipy.optimize.minimize(fun2_xy, x0, args=(ct, bt), options={'disp' : True})
    d1 = time.time()
    print('time to optimise:', d1-d0, 's') 

    # print the features with the highest error:
    print('features, highest error --> lowest error:')
    err = err_xy(res.x, ct, bt)
    i = np.argsort(err)[::-1]
    for ii in i :
        print(fes[ii], np.sqrt(err[ii]/2.))
    
    x_n, y_n, a_k, b_k = res.x[:NN], res.x[NN:2*NN], res.x[2*NN:2*NN + K**2 -2], res.x[2*NN + K**2 - 2:]
    a_k = np.concatenate(([0,], a_k[:K-1], [0,], a_k[K-1:])).reshape((K, K))
    b_k = np.concatenate(([0,0], b_k)).reshape((K, K))

    print(res)
    print('\nss poly coeffs:')
    print(a_k)
    
    print('\nfs poly coeffs:')
    print(b_k)

    pos_out = pos.copy()
    pos_out[:,1] = x_n
    pos_out[:,2] = y_n
    
    i, j = np.ogrid[-1:1:(xmax-xmin)*1j, -1:1:(ymax-ymin)*1j]
    uss = polyval2d(i, j, a_k)
    ufs = polyval2d(i, j, b_k)
    
    # generate the new feature positions
    # I_m(xj) = I_n(xi)
    # O(xj - x_m + u(xj)) = O(xi - x_n + u(xi))
    # xj + ux(x_j, y_j) = x_m - x_n + x_i + ux(x_i, y_i)
    # so we need the inverse of x - ux and y - uy 
    #fes_out = np.empty_like(fes)
    #for k, fe in enumerate(fes):
    #    xi, yi = fe[1:3]
    #    xj, yj = fe[4:6]
    #    uss_t  = i - uss
    #    ufs_t  = j - ufs
        
    #    ind = np.argmin(np.abs(uss_t - x


    return uss, ufs, pos_out

if __name__ == '__main__':
    args, params = parse_cmdline_args()

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
    #data = np.array([f['/entry_1/data_1/data'][fi][ROI[0]:ROI[1], ROI[2]:ROI[3]] for fi in good_frames])
    
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

    # positions
    # ------------------
    pos = f[group + '/pix_positions'][()]
    f.close()
    
    def get_frame(index):
        f = h5py.File(args.filename, 'r')
        # apply mask and divide by whitefield
        frame = f['entry_1/data_1/data'][index, ROI[0]:ROI[1], ROI[2]:ROI[3]] 
        #frame = frame.astype(np.float64) / W
        frame *= mask
        frame -= ~mask
        f.close()
        return frame
    
    print(pos)
    inds = pos[:, 0]
    frames = np.array([get_frame(i) for i in inds])
        
    if params['build_atlas']['polyorder'] is None :
        # build the atlas
        atlas  = feature_matching.build_atlas(frames, pos)
            
    else :
        uss, ufs, pos = calculate_positions_distortions(pos, h5_features, ROI, params['build_atlas']['polyorder'])
        # regularise
        i, j = np.ogrid[-1:1:W.shape[0]*1J, -1:1:W.shape[1]*1J]
        exp  = np.exp(-(i**2 + j**2)/(2. * 0.5**2)) * W
        atlas  = feature_matching.build_atlas_distortions(frames * exp, W*exp, pos[:, 1:], np.rint(uss).astype(np.int), np.rint(ufs).astype(np.int))
    
    translations, basis_vectors = calculate_xy_positions(pos, \
                                      params['build_atlas']['fs_range'], z, pix_size, data_len)
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

    key = params['build_atlas']['h5_group']+'/pix_positions_dist'
    if key in g :
        del g[key]
    g[key] = pos

    key = params['build_atlas']['h5_group']+'/pixel_shifts'
    if key in g :
        del g[key]
    shape = f['entry_1/data_1/data'].shape[1:]
    u = np.zeros((2,) + shape, dtype=np.float)
    u[0, ROI[0]:ROI[1], ROI[2]:ROI[3]] = uss
    u[1, ROI[0]:ROI[1], ROI[2]:ROI[3]] = ufs
    g[key] = u

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
