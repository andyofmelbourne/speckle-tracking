import speckle_tracking as st
import h5py
import numpy as np
import pyqtgraph as pg
import scipy.signal
from scipy.ndimage import filters 


# extract data
f = h5py.File('siemens_star.cxi', 'r')

data  = f['/entry_1/data_1/data'][()]
basis = f['/entry_1/instrument_1/detector_1/basis_vectors'][()]
z     = f['/entry_1/instrument_1/detector_1/distance'][()]
x_pixel_size = f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]
y_pixel_size = f['/entry_1/instrument_1/detector_1/y_pixel_size'][()]
wav          = f['/entry_1/instrument_1/source_1/wavelength'][()]
translations = f['/entry_1/sample_3/geometry/translation'][()]

mask  = f['results/mask'][()]

W   = f['results/whitefield'][()]
roi = f['results/roi'][()]
z1  = f['results/z1'][()]
dz  = f['results/dz'][()]
f.close()

#mask  = st.make_mask(data)

# estimate the whitefield
#W = st.make_whitefield(data, mask)

# estimate the region of interest
#roi = st.guess_roi(W)

# fit thon rings
#z1, dz, res = st.fit_defocus(data, x_pixel_size, y_pixel_size, z, wav, mask, W, roi)
print(z1, dz)

pixel_map, pixel_map_inv, dxy = st.make_pixel_map(z, z1, dz, roi, x_pixel_size, y_pixel_size, data.shape[1:])
print('pixel_map.shape', pixel_map.shape)
print('mask.shape', mask.shape)

# mask the area outside of the roi
t = mask[roi[0]:roi[1], roi[2]:roi[3]].copy()
mask.fill(False)
mask[roi[0]:roi[1], roi[2]:roi[3]] = t


dij_n = st.make_pixel_translations(translations, basis, dxy[0], dxy[1])

"""
sig = 20.
c   = [(roi[1]-roi[0])//2 + roi[0], (roi[3]-roi[2])//2 + roi[2]]
i0  = c[0] + 20
j0  = c[1] + 20
ii, jj = (np.arange(data.shape[1])-i0), (np.arange(data.shape[2])-j0)
sig = np.outer(np.exp(-ii**2 / (2. * sig**2)), np.exp(-jj**2 / (2. * sig**2)))
"""

I, n0, m0 = st.make_object_map(data, mask, W, dij_n, pixel_map)

"""
pixel_map2, errors, overlaps = st.update_pixel_map(
                                  data, mask, W, I, pixel_map, 
                                  n0, m0, dij_n, search_window=10,
                                  window=2)

I2, n0, m0 = st.make_object_map(sig*data, mask, sig*W, dij_n, pixel_map2)
"""
"""
Mss, Mfs = (z + dz)/(z1 + dz), (z - dz)/(z1 - dz)

dx = dy = min(x_pixel_size / Mss, y_pixel_size / Mfs)

shape = data.shape[1:]
i, j = np.arange(shape[0]), np.arange(shape[1])
i, j = np.meshgrid(i, j, indexing='ij')

pixel_map = np.array([x_pixel_size * (i - roi[0]) / (dx * Mss), 
                      y_pixel_size * (j - roi[2]) / (dy * Mfs)]) 
pixel_map = np.rint(pixel_map).astype(np.int)


rois = (slice(roi[0], roi[1], 1), slice(roi[2], roi[3], 1))
m = np.rint(pixel_map).astype(np.int)
n = np.rint(pixel_map_inv).astype(np.int)
a = np.zeros( (m[0][rois].max()+1, m[1][rois].max()+1), dtype=np.float)

#rois = (slice(roi[0], roi[1], 1), slice(roi[2], roi[3], 1))
a[m[0][rois], m[1][rois]] = data[0][rois]

b = data[0][n[0], n[1]]
"""


"""
# generate the x shifts
z2 = z - z2
z1ss = z1 + dz
z1fs = z1 - dz
gss = -z2 / z1ss
gfs = -z2 / z1fs

shape = data.shape[1:]
x, y = np.arange(shape[0]), np.arange(shape[1])

# define x=0 to be the centre of the roi
x   -= x[(roi[1]-roi[0])//2] 
y   -= y[(roi[3]-roi[2])//2] 

x   *= x_pixel_size 
y   *= y_pixel_size

x -= gss * x
y -= gfs * y

x, y = np.meshgrid(x, y, indexing='ij')

# mapping from projected image to object view
# O(x) = I(x_map)
# where I_ij = I(i * x_pixel_size, j * y_pixel_size)
x_map = np.array([x, y]) 

# map the translations onto the fs / ss axes
dx_D = np.array([np.dot(basis[i], translations[i]) for i in range(len(basis))])
dx_D[:, 0] /= x_pixel_size
dx_D[:, 1] /= y_pixel_size

# offset the translations so that the centre position is at the centre of the object
dx_D[:, 0] -= np.mean(dx_D[:, 0])
dx_D[:, 1] -= np.mean(dx_D[:, 1])

# Now we should have
# O[x] = I_n[x_map + dx_D[n]]

# convert to pixel shifts : 
# this 
"""
