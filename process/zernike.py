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

import cmdline_config_cxi_reader
import optics

import numpy as np
import scipy.constants as sc

def variance_minimising_subtraction(f, g):
    """
    find min(f - a * g)|_a
    """
    fm = np.mean(f)
    gm = np.mean(g)
    a = np.sum( (g - gm)*(f - fm) ) / np.sum( (g - gm)**2 )
    return a


def get_focus_probe(P):
    # zero pad
    P2 = np.zeros( (2*P.shape[0], 2*P.shape[1]), dtype=P.dtype)
    P2[:P.shape[0], :P.shape[1]] = P
    P2 = np.roll(P2, P.shape[0]//2, 0)
    P2 = np.roll(P2, P.shape[1]//2, 1)
     
    # real-space probe
    P2 = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(P2)))
    return P2


def calculate_Zernike_coeff(phase, mask, orders, dq, basis=None, basis_grid=None, y=None, x=None):
    # Orthogonalise the Zernike polynomials 
    # -------------------------------------
    # define the x-y grid to evaluate the polynomials 
    # evaluate on our grid [2xN, 2xM] where N and M are the frame dims
    # rectangle in circle domain
    if basis is None :
        shape = phase.shape
        rat = float(shape[0])/float(shape[1])
        x   = np.sqrt(1. / (1. + rat**2))
        y   = rat * x
        dom = [-y, y, -x, x]
        roi = shape
        y_vals  = np.linspace(dom[0], dom[1], shape[0])
        x_vals  = np.linspace(dom[2], dom[3], shape[1])
        
        basis, basis_grid, y, x = optics.fit_Zernike.make_Zernike_basis(\
                                  mask.astype(np.bool), \
                                  roi = None, max_order = orders, return_grids = True, \
                                  yx_bounds = dom, test = False)
    
    # get the Zernike coefficients
    # ----------------------------
    phi = phase 
    
    dA = (y[1]-y[0])*(x[1]-x[0])
    z = [np.sum(dA * b * phi) for b in basis_grid]
    z = np.array(z)

    # get the Zernike fit in a polynomial basis
    z_poly = np.sum( [z[i] * basis[i] for i in range(len(basis))], axis=0)

    print('\n\n')
    print('Zernike coefficients')
    print('--------------------')
    print('Noll index, weight')
    for i in range(orders):
        print(i+1, z[i])
    return z, z_poly, basis, basis_grid, y, x

def get_geometric_aberrations(phase, y, x, dq, wavelen,
        remove_piston      = False,
        remove_tilt        = False,
        remove_astigmatism = False, 
        remove_defocus     = False,
        verbose            = False):
    
    # rescale y and x 
    dA = (y[1]-y[0])*(x[1]-x[0])
    qy = y/(y[1]-y[0]) * dq[0]
    qx = x/(x[1]-x[0]) * dq[1]
    qy, qx = np.meshgrid(qy, qx, indexing='ij')
    
    # find the geometric aberrations by performing 
    # a variance minimising subtraction of each of 
    # the aberration terms
    # - remove the aberrations as we go
    
    if verbose : print('\nCalculating and removing geometric aberrations:')
    if verbose : print('variance of phase:', np.var(phase))
    
    # defocus
    # -------
    phi_df = - np.pi * wavelen * 1. * (qy**2 + qx**2)
    phi_fx = - np.pi * wavelen * 1. * qx**2
    phi_fy = - np.pi * wavelen * 1. * qy**2
    defocus   = variance_minimising_subtraction(phase, phi_df)
    defocus_x = variance_minimising_subtraction(phase, phi_fx)
    defocus_y = variance_minimising_subtraction(phase, phi_fy)
    
    if remove_defocus :
        phase -= defocus * phi_df
        if verbose : print('\nRemoving defocus', defocus)
        if verbose : print('variance of phase:', np.var(phase))

    # astigmatism 
    # ---------------------
    phi_as = - np.pi * wavelen * 1. * (qx**2 - qy**2)
    astigmatism = variance_minimising_subtraction(phase, phi_as)

    if remove_astigmatism :
        phase -= astigmatism * phi_as
        if verbose : print('\nRemoving astigmatism', astigmatism)
        if verbose : print('variance of phase:', np.var(phase))

    # tilt x (or fs)
    # ---------------------
    phi_tx = -2. * np.pi * 1. * qx
    tilt_x = variance_minimising_subtraction(phase, phi_tx)
    
    if remove_tilt :
        phase -= tilt_x * phi_tx
        if verbose : print('\nRemoving tilt_x', tilt_x)
        if verbose : print('variance of phase:', np.var(phase))

    # tilt y (or ss)
    # ---------------------
    phi_ty = -2. * np.pi * 1. * qy
    tilt_y = variance_minimising_subtraction(phase, phi_ty)
    
    if remove_tilt :
        phase -= tilt_y * phi_ty
        if verbose : print('\nRemoving tilt_y', tilt_y)
        if verbose : print('variance of phase:', np.var(phase))

    # piston
    # ---------------------
    piston = np.mean(phase)
    
    if remove_piston :
        phase -= piston
        if verbose : print('\nRemoving piston', piston)
        if verbose : print('variance of phase:', np.var(phase))
    
    
    if verbose : print('\n\n')
    if verbose : print('Geometric aberrations')
    if verbose : print('---------------------')
    if verbose : print('defocus       :', defocus, '(m) (+ve is overfocus)')
    if verbose : print('defocus fs    :', defocus_x, '(m)')
    if verbose : print('defocus ss    :', defocus_y, '(m)')
    if verbose : print('astigmatism   :', astigmatism, '(m)')
    if verbose : print('tilt fs       :', tilt_x, '(rad) relative to centre of roi')
    if verbose : print('tilt ss       :', tilt_y, '(rad) relative to centre of roi')

    return phase

def make_phase(pixel_shifts, z, du, lamb, df):
    phi_y = 2 * df * du[0]**2 * np.pi / (lamb * z) * np.cumsum(pixel_shifts[0], axis=0, dtype=np.float)
    phi_x = 2 * df * du[1]**2 * np.pi / (lamb * z) * np.cumsum(pixel_shifts[1], axis=1, dtype=np.float)
    
    # this is because broadcasting is performed along the last dimension
    phi = phi_y + phi_x + phi_x[0, :]
    phi  = (phi.T + phi_y[:, 0].T).T 
    phi /= 2.
    return -phi

if __name__ == '__main__':
    # get input 
    ###########
    args, params = cmdline_config_cxi_reader.get_all('zernike', 
                   'stitch frames together to form a merged view of the sample from projection images')
    params = params['zernike']
    
    phase = np.zeros_like(params['pixel_shifts'][0])
    
    # calculate the Zernike coefficients
    # ----------------------------------
    # calcualte dq
    wavelen = sc.h * sc.c / params['energy']
    du      = np.array([params['x_pixel_size'], params['y_pixel_size']])
    dq      = du / (wavelen * params['distance'])

    phase = make_phase(params['pixel_shifts'], params['distance'], du, wavelen, params['defocus'])
    z, z_poly, basis, basis_grid, y, x = calculate_Zernike_coeff(phase, params['mask'], params['orders'], dq)
    
    # get defocus, astigmatism and tilt
    # ---------------------------------
    for i in range(5):
        phase = get_geometric_aberrations(phase, y, x, dq, wavelen, \
                remove_piston      = params['remove_piston'], \
                remove_tilt        = params['remove_tilt'], \
                remove_astigmatism = params['remove_astigmatism'], \
                remove_defocus     = params['remove_defocus'],
                verbose= i==0)
    
    # calculate the Zernike again
    # ----------------------------------
    z, z_poly, basis, basis_grid, y, x = calculate_Zernike_coeff(phase, params['mask'], 
                                                    params['orders'], dq, basis, basis_grid, y, x)
    
    # make the Zernike fit
    # ---------------------------------
    phase_zern = np.sum( [z[i] * basis_grid[i] for i in range(len(basis))], axis=0)
    
    # get the focus spot
    if params['whitefield'] is not None :
        print('\ncalculating focus...')
        pupil = params['whitefield'] * np.exp(1J * phase)
        P_focus = get_focus_probe(pupil)
        print('Real space pixel size   :', 1./ (np.array(P_focus.shape) * dq))
        print('Real space Field-of-view:', 1./ dq)

    
    # write the result 
    ##################
    out = {'probe_focus': P_focus, 
           'phase': phase, 
           'zernike_phase_fit' : phase_zern,
           'zernike_coefficients': z, 
           'zernike_basis_vectors': basis_grid}
    cmdline_config_cxi_reader.write_all(params, args.filename, out)
    
    print('display: '+params['h5_group']+'/zernike_coefficients') ; sys.stdout.flush()
    
