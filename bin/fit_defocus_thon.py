"""
for T=e^{i phi}
"""
#! /usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# make an example cxi file
# with a small sample and small aberations

import sys, os
base = os.path.join(os.path.dirname(__file__), '..')
root = os.path.abspath(base)
sys.path.insert(0, os.path.join(root, 'utils'))

import pyximport; pyximport.install()
import feature_matching as fm
import cmdline_config_cxi_reader
import cmdline_parser 

import numpy as np

def mk_reg(shape, reg):
    if reg is not None : 
        x = np.arange(shape[0]) - shape[0]//2
        x = np.exp( -x**2 / (2. * reg**2))
        y = np.arange(shape[1]) - shape[1]//2
        y = np.exp( -y**2 / (2. * reg**2))
        reg = np.outer(x, y)
    else :
        reg = 1
    return reg

def get_r_theta(shape, is_fft_shifted = True):
    i = np.fft.fftfreq(shape[0]) * shape[0]
    j = np.fft.fftfreq(shape[1]) * shape[1]
    i, j = np.meshgrid(i, j, indexing='ij')
    rs   = np.sqrt(i**2 + j**2)
    
    if is_fft_shifted is False :
        rs = np.fft.fftshift(rs)
    
    if angle_range is not None :
        ts = np.arctan2(i, j)
    return rs, ts

def radial_symetry(background, rs, mask=None):
    ########### Find the radial average
    # mask zeros from average
    if mask is None : 
        mask = np.ones(background.shape, dtype=np.bool)

    rs = rs.astype(np.int16).ravel()
    
    # get the r histogram
    r_hist = np.bincount(rs, mask.ravel().astype(np.int16))
    # get the radial total 
    r_av = np.bincount(rs, background.ravel() * mask.ravel())
    # prevent divide by zero
    nonzero = np.where(r_hist != 0)
    zero    = np.where(r_hist == 0)
    # get the average
    r_av[nonzero] = r_av[nonzero] / r_hist[nonzero].astype(r_av.dtype)
    r_av[zero]    = 0
    
    ########### Make a large background filled with the radial average
    background = r_av[rs].reshape(background.shape)
    return background, r_av

def make_thon_from_data(frames):
    a = np.zeros(frames[0].shape, dtype=np.float64)
    count = 0.
    for frame in frames :
        a     += np.real(np.fft.fftn(np.fft.ifftshift(reg * frame * (frame>0))))**2
        count += 1.
    
    a /= count
    return a

def thon_rad_av(thon, edge=10, sigma=10, rad_range=None, angle_range=None, flatten=True):
    thon_2 = thon.copy()
    
    # edge, angle, radius mask
    rad, theta = get_r_theta(thon.shape)
    mask = np.ones(thon.shape, dtype=np.bool)
    if angle_range is not None :
        mask[theta < angle_range[0]] = False
        mask[theta >=angle_range[1]] = False
    if edge is not None :
        mask[:edge,  :] = False
        mask[-edge:, :] = False
        mask[:,  :edge] = False
        mask[:, -edge:] = False
    
    rs = rad.astype(np.int16).ravel()
    
    # average over angle
    thon_av, thon_1d = radial_symetry(thon_2, rs, mask)

    env_2d   = np.ones_like(thon)
    env_1d   = np.ones_like(thon_1d)
    if flatten :
        import scipy.ndimage
        env_1d   = 2*scipy.ndimage.filters.gaussian_filter(thon_1d, sigma)
        
        env_2d   = np.ones_like(thon)
        env_2d   = env_1d[rs].reshape(thon.shape)
    
    if rad_range is not None :
        mask[rad < rad_range[0]] = False
        mask[rad >=rad_range[1]] = False
    return thon/env_2d, thon_av/env_2d, thon_1d/env_1d, rs, mask
        

def fit_thon(thon_1d, wavelength, distance, x_pixel_size, 
             y_pixel_size, defocus_range, N, rad_range=None):
    
    if rad_range is None :
        rad_range = [0, thon_1d.shape[0]]
    
    n2  = np.pi * wavelength * np.arange(thon_1d.shape[0])**2 * distance / (N**2 * x_pixel_size**2)

    def forward(z):
        return np.sin((distance - z) * n2 / z)**2
        
    import scipy.stats
    def fun(z):
        err, _ = scipy.stats.pearsonr(thon_1d[rad_range[0]:rad_range[1]], 
                                      forward(z[0])[rad_range[0]:rad_range[1]])
        return -err
    
    zs = np.linspace(defocus_range[0], defocus_range[1], 10000)
    errs = np.array([fun([z]) for z in zs])
    return forward(zs[np.argmin(errs)]), zs, errs
    

if __name__ == '__main__':
    # get input 
    ###########
    # get command line args and config
    sc  = 'fit_defocus_thon'
    des = 'fit the sample to focus distance to the Thon rings.'
    args, params = cmdline_parser.parse_cmdline_args(sc, des)
    
    if params['fit_defocus_thon']['use_existing_thon'] is not False :
        exclude = ['frames', 'whitefield']
    else :
        exclude = []
    
    # now load the necessary data
    args, params = cmdline_config_cxi_reader.get_all(sc,des,exclude)
    params = params['fit_defocus_thon']
    
    # Do Stuff
    ##########
    if params['use_existing_thon'] is False :
        # set masked pixels to negative 1
        mask              = params['mask'].astype(np.bool)
        params['frames']  = params['frames'][params['good_frames']].astype(np.float64)
        for i in range(len(params['frames'])):
            params['frames'][i][~mask] = -1
        
        if len(params['whitefield'].shape) == 3 :
            for i in range(len(params['whitefield'])):
                params['whitefield'][i][~mask]    = -1 
        else :
            params['whitefield'][~mask]    = -1 
        
        # add a regularization factor
        shape = params['whitefield'].shape[-2:]
        reg   = mk_reg(shape, params['reg'])
        
        # calculate 2D thon rings from data
        thon = make_thon_from_data(params['frames']/params['whitefield'])
    else :
        thon = params['use_existing_thon']
    
    # calculate the radial average within roi
    if params['angle_range'] is not None :
        angle_range = np.array(params['angle_range'])*np.pi

    thon_flat, thon_av, thon_1d, rs, mask = thon_rad_av(thon, params['edge_mask'],
                                      params['smooth'], params['pix_range'], 
                                      angle_range, flatten=True)
    
    t_fit, zs, errs = fit_thon(thon_1d, params['wavelength'], 
                     params['distance'], params['x_pixel_size'], 
                     params['y_pixel_size'], params['defocus_range'], 
                     thon.shape[0], params['pix_range'])


    # show the flattened thon rings + a wedge showing the fit
    thon_show = t_fit[rs].reshape(thon.shape)*mask
    thon_show[~mask] = thon_flat[~mask]
    thon_show[thon_show>2] = 2
    thon_show = np.fft.fftshift(thon_show)
    
    #import pyqtgraph as pg
    #plot = pg.plot(thon_1d)
    #plot.plot(t_fit)
    #pg.plot(zs, errs)
    #pg.show(np.fft.fftshift(thon_show))
    #print('defocus:', zs[np.argmin(errs)], 'meters')
    
    # write the result 
    ##################
    out = {'thon': thon, 'thon_with_fit': thon_show, 'thon_radial_average': np.fft.fftshift(thon_av),
           'fits': np.array([thon_1d, t_fit]), 
           'errs': np.array([errs, zs]), 'defocus' : zs[np.argmin(errs)]}
    cmdline_config_cxi_reader.write_all(params, args.filename, out)
    
