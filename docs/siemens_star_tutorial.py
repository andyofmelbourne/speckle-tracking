import speckle_tracking as st
import h5py
import numpy as np
output_group = 'speckle_tracking/'


"""
# extract data
f = h5py.File('siemens_star.cxi', 'r')

data  = f['/entry_1/data_1/data'][()].astype(np.float32)
basis = f['/entry_1/instrument_1/detector_1/basis_vectors'][()]
z     = f['/entry_1/instrument_1/detector_1/distance'][()]
x_pixel_size = f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]
y_pixel_size = f['/entry_1/instrument_1/detector_1/y_pixel_size'][()]
wav          = f['/entry_1/instrument_1/source_1/wavelength'][()]
translations = f['/entry_1/sample_1/geometry/translation'][()]
f.close()

output_group = 'speckle_tracking/'

mask  = st.make_mask(data)
f = h5py.File('siemens_star.cxi')
f[output_group+'mask'] = mask
f.close()

W = st.make_whitefield(data, mask)
f = h5py.File('siemens_star.cxi')
f[output_group+'whitefield'] = W
f.close()

roi = st.guess_roi(W)
f = h5py.File('siemens_star.cxi')
f[output_group+'roi'] = np.array(roi)
f.close()


defocus, dz, res = st.fit_defocus(
                      data,
                      x_pixel_size, y_pixel_size,
                      z, wav, mask, W, roi)


f = h5py.File('siemens_star.cxi')
f[output_group+'defocus'] = defocus
f[output_group+'defocus_fs'] = defocus - dz
f[output_group+'defocus_ss'] = defocus + dz
f[output_group+'fit_defocus/thon_rings'] = res['thon_display']
f[output_group+'fit_defocus/beta_on_alpha'] = res['bd']
f.close()

"""
f = h5py.File('siemens_star.cxi', 'r')
x_pixel_size = f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]
y_pixel_size = f['/entry_1/instrument_1/detector_1/y_pixel_size'][()]
wav          = f['/entry_1/instrument_1/source_1/wavelength'][()]
z    = f['/entry_1/instrument_1/detector_1/distance'][()]
mask = f[output_group+'mask'][()]
W    = f[output_group+'whitefield'][()]
roi  = f[output_group+'roi'][()]
defocus = f[output_group+'defocus'][()]
defocus_fs = f[output_group+'defocus_fs'][()]
defocus_ss = f[output_group+'defocus_ss'][()]
f.close()

roi2 = [0, W.shape[0], 0, W.shape[1]]

dz = defocus - defocus_fs

pm0, pixel_map_inv, dxy = st.make_pixel_map(
                                   z, defocus, dz, roi2,
                                   x_pixel_size, y_pixel_size,
                                   W.shape)

"""
dij_n = st.make_pixel_translations(translations, basis, dxy[0], dxy[1])

O, n0, m0 = st.make_object_map(data, mask, W, dij_n, pm0, subpixel=False)

pixel_map, map_mask, res = st.update_pixel_map(data, mask, W, O, pm0, n0, m0, dij_n, 
                           roi=roi, guess=True)

pm1 = pixel_map.copy()

O, n0, m0 = st.make_object_map(data, map_mask, W, dij_n, pixel_map, subpixel=True)

error_total, error_frame, error_pixel = st.calc_error(
                                        data, map_mask, W, dij_n, O, 
                                        pixel_map, n0, m0, subpixel=True)

# now update bad pixels
from speckle_tracking.update_pixel_map import update_pixel_map_opencl
u, v = np.where(error_pixel/W**2 > 0.04)
out, res = update_pixel_map_opencl(data, mask, W, O, pixel_map,
                                          n0, m0, dij_n, roi, False, 1.,
                                          [100, 100], u, v)
pixel_map[0][u, v] = out[0]
pixel_map[1][u, v] = out[1]



write = {'object_map': O, 
         'n0': n0, 'm0': m0, 
         'pixel_map': pixel_map, 
         'map_mask': map_mask, 
         'error_pixel': error_pixel}

f = h5py.File('siemens_star.cxi')
for k in write.keys():
    key = output_group+k
    if key in f :
        del f[key]
    f[key] = write[k]
f.close()
"""

og = 'speckle_tracking/'
f = h5py.File('siemens_star.cxi', 'r')
pm   = f[og+'pixel_map'][()]
mask = f[og+'mask'][()]
f.close()

pm -= pm0

from scipy.ndimage.filters import gaussian_filter
pm[0] = gaussian_filter(pm[0], 2., mode='constant')
pm[1] = gaussian_filter(pm[1], 2., mode='constant')

#from speckle_tracking.utils import integrate
#from speckle_tracking.utils import P
#from speckle_tracking.utils import P
# now integrate the phase derivatives
#phi = integrate(pm[0][roi[0]:roi[1],roi[2]:roi[3]] , pm[1][roi[0]:roi[1],roi[2]:roi[3]], np.ones_like(mask)[roi[0]:roi[1],roi[2]:roi[3]], tol=1e-10, maxiter=20)
#phi = integrate(pm[0] , pm[1], np.ones_like(mask), tol=1e-10, maxiter=20)
#phi = integrate(pm[0][roi[0]:roi[1],roi[2]:roi[3]] , pm[1][roi[0]:roi[1],roi[2]:roi[3]], np.ones_like(mask)[roi[0]:roi[1],roi[2]:roi[3]], tol=1e-10, maxiter=5)

mask[:, 0] = False
mask[0, :] = False
dss = pm[0]
dfs = pm[1]
s = dss.shape
dfx = np.zeros((s[0], s[1]+1), dtype=float)
dfy = np.zeros((s[0]+1, s[1]), dtype=float)
mx  = np.zeros(dfx.shape, dtype=np.bool)
my  = np.zeros(dfy.shape, dtype=np.bool)
mx[:, :-1] = mask
my[:-1, :] = mask
dfx[:, :-1] = dss
dfy[:-1, :] = dfs
dfx *= mx
dfy *= my
norm = np.sum(dss**2 + dfs**2)
f = np.zeros((s[0]+1, s[1]+1), dtype=float)


# mx  :  oxxxoooxo 
# seg : x1xx222233 
my_seg = np.zeros((mask.shape[0]+1, mask.shape[1]+1), dtype=np.int32)
val = 0
for i in range(0, my.shape[0]):
    for j in range(0, my.shape[1]):
        # if the last value is False and this value is True then increment:
        if (j>0) and my[i, j] and (my[i, j-1] == False):
            val+=1
            my_seg[i,j] = val
        
        # if the mask is True set to val:
        if my[i,j] :
            my_seg[i,j+1] = val

mx_seg = np.zeros((mask.shape[0]+1, mask.shape[1]+1), dtype=np.int32)
val = 0
for j in range(0, mx.shape[1]):
    for i in range(0, mx.shape[0]):
        # if the last value is False then increment:
        if (i>0) and mx[i, j] and (mx[i-1, j] == False):
            val+=1
            mx_seg[i,j] = val
        
        # if the mask is True set to val:
        if mx[i, j] :
            mx_seg[i+1,j] = val

def diff_inv_2d(da, a0=None, axis=0):
    shape        = list(da.shape)
    shape[axis] += 1
    out          = np.zeros(tuple(shape), dtype=float)
    if a0 is None :
        if axis == 0 :
            a0 = np.zeros((shape[1],))
        else :
            a0 = np.zeros((shape[0],))
        
    if axis == 0 :
        out[0,  :] = a0
        out[1:, :] = da
    elif axis == 1 :
        out[:,  0] = a0
        out[:, 1:] = da
    return np.cumsum(out, axis=axis)

def mean_adjust(g,h,m_seg):
    totals = np.bincount(m_seg.ravel(), (g-h).ravel())
    counts = np.bincount(m_seg.ravel())
    means  = totals / counts.astype(np.float)
    means  = means[m_seg.ravel()].reshape(m_seg.shape)
    
    h[m_seg>0] += means[m_seg>0]
    h[m_seg==0] = g[m_seg==0]

    #assert(np.allclose(np.bincount(m_seg.ravel(), h.ravel()), np.bincount(m_seg.ravel(), g.ravel())))
    return h


def P(g, df, m_seg, axis):
    if axis == 0 :
        h  = diff_inv_2d(df, a0 = None, axis=0)
    elif axis == 1 :
        h = diff_inv_2d(df, a0 = None, axis=1)
    
    h  = mean_adjust(g, h, m_seg)
    return h

#fy = P(f, dfy, my_seg, 1)
#fx = P(f, dfx, mx_seg, 0)
#print(np.sum( my*(np.diff(fy, axis=1) - dfy)**2 ))
#print(np.sum( mx*(np.diff(fx, axis=0) - dfx)**2 ))
fx  = np.cumsum(dfx, axis=0)
fx  = (fx.T - fx[:, fx.shape[1]//2]).T
fy  = np.cumsum(dfy, axis=1)
fy -= fy[fy.shape[0]//2, :]
f[1:,:] = fx
f[:,1:]+= fy
f = f / 2.

fs = [f.copy()]

for j in range(2):
    maxiter = 500
    for i in range(maxiter):
        fx = P(f, dfx, mx_seg, 0)
        fy = P(f, dfy, my_seg, 1)
        #f  = f - 1.5*(f - (fx+fy)/2.)
        f = (fx+fy)/2.
        if j == 0 :
            f = gaussian_filter(f, 4., mode='constant')
        # print(np.sum( my*(np.diff(f, axis=1) - dfy)**2 ))
        #print(np.sum( (f - P(f, dfx, mx_seg, 0))**2 ))
        print(i, round(np.sum(mask*(np.diff(f, axis=0)[:, :-1] - pm[0])**2)), round(np.sum( (f - fx)**2 )))
        #print(np.sum(mask*(np.diff(f, axis=1)[:-1, :] - pm[1])**2))
        #print(np.sum( (f - fx)**2 ))
        #print(np.sum( (f - fy)**2 ))
        if i % (maxiter//100) == 0 :
            fs.append(f.copy())

fs.append(f.copy())
