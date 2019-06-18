import speckle_tracking as st
import h5py
import numpy as np
og = 'speckle_tracking/'

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


#---------------------------
# intialise
#---------------------------
mask  = st.make_mask(data)

W = st.make_whitefield(data, mask)

roi = st.guess_roi(W)

dz, res = st.fit_defocus(
             data,
             x_pixel_size, y_pixel_size,
             z, wav, mask, W, roi)

pixel_map, uinv, dxy = st.make_pixel_map(
                  z, dz, res['astigmatism'], 
                  roi, x_pixel_size, y_pixel_size,
                  W.shape)

dij_n = st.make_pixel_translations(
           translations, 
           basis, dxy[0], dxy[1])

O, n0, m0 = st.make_object_map(
               data, mask, W, dij_n, pixel_map)

#---------------------------
# Refine
#---------------------------
us   = [pixel_map.copy()]
Os   = [O.copy()]
dijs = [dij_n.copy()]

for i in range(10):
    pixel_map, res = st.update_pixel_map(
                data, mask, W, O, pixel_map, n0, m0, dij_n, 
                clip = [-40, 40],
                fill_bad_pix = True, 
                integrate = True, 
                quadratic_refinement = True)
    
    O, n0, m0  = st.make_object_map(data, mask, W, dij_n, pixel_map, subpixel=True)
    dij_n, res = st.update_translations(data, mask, W, O, pixel_map, n0, m0, dij_n)
    us.append(pixel_map.copy())
    Os.append(O.copy())
    dijs.append(dij_n.copy())

#---------------------------
# Convert to physical units
#---------------------------
# get angles 
# O[ss]   = pixel_map[0 + i*J + j] - di - dij_n[n*2 + 0] + n0;

# O[pixel_map[i, j] + n0] = I[i, j]
# O[pixel_map[i, j] + n0] = O( dx (pixel_map[i, j] + n0) ) 
#                         = I(i u)
# I(x)   = O(x  - wav z / 2 pi grad(x))
# I(i u) = O(iu - wav z / 2 pi grad(i u))
# iu - wav z / 2 pi grad(i u) = dx (pixel_map[i, j] + n0)
# iu - dx (pixel_map[i, j] + n0) = wav z / 2 pi grad(i u) 
# theta(i u) = iu / z - dx (pixel_map[i, j] + n0) / z
ij = np.array(np.indices(W.shape))
theta = np.zeros_like(pixel_map)
theta[0] = (ij[0] * x_pixel_size - dxy[0] * (pixel_map[0] + n0))/z
theta[1] = (ij[1] * y_pixel_size - dxy[1] * (pixel_map[1] + m0))/z

# remove tilt
def variance_minimising_subtraction(f, g):
    """
    find min(f - a * g)|_a
    """
    fm = 0# np.mean(f)
    gm = np.mean(g)
    a = np.sum( (g - gm)*(f - fm) ) / np.sum( (g - gm)**2 )
    return a

# preferentialy remove tilts from the centre of the roi
cx   = (roi[1]-roi[0])/2 + roi[0]
cy   = (roi[3]-roi[2])/2 + roi[2]
sig  = [(roi[1]-roi[0])/8., (roi[3]-roi[2])/8.]
gaus = np.exp( -(ij[0]-cx)**2 / (2. * sig[0]**2) \
               -(ij[1]-cy)**2 / (2. * sig[1]**2))

# remove tilt and defocus
#gaus = 1
for i in range(10000):
    theta[0] -= np.mean(gaus * theta[0])
    theta[1] -= np.mean(gaus * theta[1])
    a = variance_minimising_subtraction(gaus * theta, gaus * ij)
    print(a)
    theta -= a * ij

# remove offset tilt defocus from phase
# get propagation profile

#---------------------------
# Write
#---------------------------
write = {'object_map': O, 
         'n0': n0, 'm0': m0, 
         'pixel_map': pixel_map,
         'pixel_translations': dij_n,
         }

with h5py.File('siemens_star.cxi') as f:
    for k in write.keys():
        key = og+k
        if key in f :
            del f[key]
        f[key] = write[k]


