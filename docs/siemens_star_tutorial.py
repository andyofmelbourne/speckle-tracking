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

mask  = st.make_mask(data)

W = st.make_whitefield(data, mask)

roi = st.guess_roi(W)

dz, res = st.fit_defocus(
             data,
             x_pixel_size, y_pixel_size,
             z, wav, mask, W, roi)

u0, uinv, dxy = st.make_pixel_map(
                  z, dz, res['astigmatism'], 
                  roi, x_pixel_size, y_pixel_size,
                  W.shape)

dij_n = st.make_pixel_translations(
           translations, 
           basis, dxy[0], dxy[1])

O, n0, m0 = st.make_object_map(
               data, mask, W, dij_n, u0)

#u, res = st.update_pixel_map(
#            data, mask, W, O, u,
#            n0, m0, dij_n)

ss, fs = np.indices(u.shape[1:])

u, res = st.update_pixel_map_opencl(
         data, mask, W, O, u0,
         n0, m0, dij_n, False, 1.,
         [68,55], ss.ravel(), fs.ravel())

u = u.reshape((2,) + W.shape)

phase, res = st.integrate_pixel_map(u, W, wav, z-dz)

O, n0, m0 = st.make_object_map(
               data, mask, W, 
               dij_n, u, subpixel=True)

e_total, e_frame, e_pixel = st.calc_error(
                               data, mask, W, dij_n, O, 
                               u, n0, m0, subpixel=True)

write = {'object_map': O, 
         'n0': n0, 'm0': m0, 
         'pixel_map': u, 
         'phase': phase, 
         'error_pixel': e_pixel,
         'error_total': e_total,
         'error_frame': e_frame}

f = h5py.File('siemens_star.cxi')
for k in write.keys():
    key = og+k
    if key in f :
        del f[key]
    f[key] = write[k]
f.close()


#read = {}
#f = h5py.File('siemens_star.cxi', 'r')
#for k in write.keys():
#    key = og+k
#    read[key] = f[key][()]
#f.close()

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
                                          n0, m0, dij_n, False, 1.,
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

og = 'speckle_tracking/'
f = h5py.File('siemens_star.cxi', 'r')
pm   = f[og+'pixel_map'][()]
mask = f[og+'mask'][()]
W = f[og+'whitefield'][()]
f.close()

pm -= pm0

from speckle_tracking.utils import integrate
phase, res = integrate(pm[0], pm[1], W)
"""


