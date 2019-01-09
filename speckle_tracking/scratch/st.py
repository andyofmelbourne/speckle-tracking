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
f.close()

#mask  = st.make_mask(data)

# estimate the whitefield
W = st.make_whitefield(data, mask)

# estimate the region of interest
roi = st.guess_roi(W)

# fit thon rings
z1, dz, res = st.fit_defocus(data, x_pixel_size, y_pixel_size, z, wav, mask, W, roi)
print(z1, dz)

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

