#! /usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# make an example cxi file
# with a small sample and small aberations

import sys, os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root, 'utils'))

import config_reader
import cmdline_parser

import h5py
import scipy.misc
import numpy as np


def add_poisson_noise(frames, Pd, n):
    # normalise
    norm   = np.sum(np.abs(Pd)**2) 
    frames = frames / norm
    
    # Poisson sampling
    frames = np.random.poisson(lam = float(n) * frames).astype(np.int)
    return frames

def make_object(**kwargs):
    # transmission of the object: 0 --> 255
    O0 = scipy.misc.ascent().astype(np.float)

    # scale: 0.8 --> 1.
    O0 = O0 * 0.2 / O0.max() + 0.8
    
    # padd
    O = np.ones((round(1.5*O0.shape[0]), round(1.5*O0.shape[1])), dtype=np.float)
    s = [round(O0.shape[0]/4.)-1, round(O0.shape[1]/4.)-1]
    O[s[0] :s[0] + O0.shape[0], s[1] : s[1] + O0.shape[1]] = O0
    
    # sample plane sampling
    dx  = kwargs['pix_size'] * kwargs['defocus'] / kwargs['det_dist']
    
    # interpolate
    #############
    # desired grid 
    yd, xd = np.arange(0, kwargs['o_size'], dx), np.arange(0, kwargs['o_size'], dx)
    
    # current x, y values
    y, x = np.linspace(0, kwargs['o_size'], O.shape[0]), np.linspace(0, kwargs['o_size'], O.shape[1])
    
    # interpolate onto the new grid
    from scipy.interpolate import RectBivariateSpline
    rbs = RectBivariateSpline(y, x, O, bbox = [y.min(), y.max(), x.min(), x.max()], s = 1., kx=1, ky=1)
    O = rbs(yd, xd)

    # smooth a little
    from scipy.ndimage import gaussian_filter
    O = gaussian_filter(O, 1.0, mode='constant') 
    return O

def make_probe(**kwargs):
    shape = (int(kwargs['det_shape'][0]), int(kwargs['det_shape'][1]))
    
    # aperture 
    X = kwargs['det_dist'] * kwargs['lens_size'] / kwargs['focal_length']
    X_pix = X / kwargs['pix_size']

    # centre the aperture in the detector
    roi = [0, shape[0], 0, shape[1]]
    if X_pix < shape[0]:
        roi[0] = round((shape[0]-X_pix)/2)
        roi[1] = round(shape[0] - (shape[0]-X_pix)/2)
    
    if X_pix < shape[1]:
        roi[2] = round((shape[1]-X_pix)/2)
        roi[3] = round(shape[1] - (shape[1]-X_pix)/2)

    # probe
    P = np.zeros(shape, dtype=np.float) 
    P[roi[0]:roi[1], roi[2]:roi[3]] = 1.
    
    # smooth the edges 
    from scipy.ndimage import gaussian_filter
    P = gaussian_filter(P, 2.0, mode='constant') + 0J
    
    back_prop = make_prop(P.shape, kwargs['det_dist'], kwargs['defocus'], kwargs['pix_size'], kwargs['energy'], inverse=True)
    
    # real-space probe
    Ps = back_prop(P)
    return Ps, P

def _make_frames(O, Ps, forward_prop, pos):
    # in pixels
    y_n, x_n = pos

    i, j = np.indices(Ps.shape)
    
    # keep y_n and x_n in array bounds
    y_n -= y_n.max()
    x_n -= x_n.max()

    # make frames 
    frames = []
    for y, x in zip(y_n, x_n):
        ss = np.rint(i - y).astype(np.int)
        fs = np.rint(j - x).astype(np.int)
        frame = forward_prop(Ps * O[ss, fs])
        frame = np.abs(frame)**2
        frames.append(frame) 
    
    return frames

def make_prop(shape, z, df, du, en, inverse=False):
    """
    wave_det = IFFT[ FFT[wav_sample] * e^{-i \pi \lambda z_eff (q * z / df)**2} ]
    where q_n = n z/N df du, x_n = n du
    """
    # wavelength
    from scipy import constants as sc
    wav = sc.h * sc.c / en
    
    dx    = du 
    z_eff = df * (z-df) / z

    # check if the exponential is adequately sampled
    df_min = z**2 / (np.min(shape) * du**2 / (2*wav) + z)
    if df < df_min :
        print("WARNING: defocus", df,
              " is less than the minimum value:", 
              df_min, " required for adequate sampling of the propagator.")
    
    qi, qj = np.fft.fftfreq(shape[0], dx), np.fft.fftfreq(shape[1], dx)
    qi, qj = np.meshgrid(qi, qj, indexing='ij')
    q2 = (z/df)**2 * (qi**2 + qj**2)
    ex = np.exp(-1.0J * np.pi * wav * z_eff * q2)
    
    if inverse :
        ex = ex.conj()
    
    prop = lambda x : np.fft.ifftn(np.fft.fftn(x) * ex)
    return prop

def make_frames(**kwargs):
    """
    psi_s(x)_n = Ps(x) x O(x-x_n), x_n = sample shift 
    x_n = -n (Xo - Xp)/N, 
    N = no. of positions, Xo = object size, Xp = probe size
    """
    # make the sample positions
    #--------------------------
    O = make_object(**kwargs)
    
    # make the probe
    #---------------
    Ps, Pd = make_probe(**kwargs)
    
    # make the sample positions
    #--------------------------
    dx  = kwargs['pix_size'] * kwargs['defocus'] / kwargs['det_dist']
    Xp  = np.array(Ps.shape) * dx # probe dimensions
    
    if kwargs['o_size'] < np.max(Xp) :
        raise ValueError('Error: o_size is less than the probe size... Make o_size bigger than '+str(np.max(Xp)))
    
    y_n, ystep = np.linspace(0, -(kwargs['o_size'] - Xp[0]), kwargs['ny'], retstep=True)
    x_n, xstep = np.linspace(0,  (kwargs['o_size'] - Xp[1]), kwargs['nx'], retstep=True)
    
    y_n,  x_n  = np.meshgrid(y_n, x_n, indexing='ij')
    y0_n, x0_n = y_n.flatten(), x_n.flatten()
    
    # add random offset
    y_n = y0_n + (0.5-np.random.random(y0_n.shape)) * kwargs['rand_offset'] * ystep
    x_n = x0_n + (0.5-np.random.random(x0_n.shape)) * kwargs['rand_offset'] * xstep
     
    # keep y_n and x_n in array bounds
    y_n[y_n>0] = 0
    x_n[x_n<0] = 0
    
    y_n[ y_n < -(kwargs['o_size'] - Xp[0])] = -(kwargs['o_size'] - Xp[0])
    x_n[ x_n >  (kwargs['o_size'] - Xp[1])] =  (kwargs['o_size'] - Xp[1])
    
    # make the forward propagator
    #----------------------------
    forward_prop = make_prop(Ps.shape, kwargs['det_dist'], kwargs['defocus'], kwargs['pix_size'], kwargs['energy'])
     
    # make the frames
    #----------------
    frames = _make_frames(O, Ps, forward_prop, (-y_n/dx, x_n/dx))
     
    # Poisson sampling
    #-----------------
    frames = add_poisson_noise(frames, Pd, kwargs['photons'])
    return Pd, Ps, O, (x0_n, y0_n), frames

def h5w(f, key, data):
    if key in f :
        del f[key]
    f[key] = data


def write_data(filename, Pd, pos, frames, **params):
    """
    """
    pos_xyz = np.zeros((len(frames), 3), dtype=np.float)
    pos_xyz[:, 0] = pos[0]
    pos_xyz[:, 1] = pos[1]
    pos_xyz[:, 2] = params['defocus']

    basis_vec = np.zeros((len(frames), 2, 3), dtype=np.float)
    basis_vec[:, 0] = np.array([0, -params['pix_size'], 0])
    basis_vec[:, 1] = np.array([params['pix_size'],  0, 0])
    
    f = h5py.File(filename, 'a')
    h5w(f, '/entry_1/data_1/data', frames.astype(np.uint32))
    h5w(f, '/entry_1/instrument_1/detector_1/distance', params['det_dist'])
    h5w(f, '/entry_1/instrument_1/detector_1/x_pixel_size', params['pix_size'])
    h5w(f, '/entry_1/instrument_1/detector_1/y_pixel_size', params['pix_size'])
    h5w(f, '/entry_1/instrument_1/source_1/energy', params['energy'])
    h5w(f, '/entry_1/instrument_1/detector_1/basis_vectors', basis_vec)
    h5w(f, '/entry_1/sample_3/geometry/translation', pos_xyz)
    h5w(f, '/process_2/powder', np.sum(frames, axis=0))
    h5w(f, '/process_3/mask', np.ones_like(frames[0], dtype=np.bool))

    whitefield = np.abs(Pd)**2 / np.sum(np.abs(Pd)**2) * params['photons']
    h5w(f, '/process_2/whitefield', whitefield.astype(np.uint32))
    f.close()

if __name__ == '__main__':
    args, params = cmdline_parser.parse_cmdline_args('forward_sim', 'generate an example cxi file for testing')
    
    # make the frames
    Pd, Ps, O, pos, frames = make_frames(**params)
    
    # write the frames
    write_data(args.filename, Pd, pos, frames, **params)

    # write the result 
    ##################
    config_reader.write_h5(args.filename, params['h5_group'], 
                           {'O' : O.astype(np.complex128),
                            '/Pupil' : Pd, 
                            'good_frames' : np.arange(len(frames))})
    print('display: '+params['h5_group']+'/O') ; sys.stdout.flush()
    
