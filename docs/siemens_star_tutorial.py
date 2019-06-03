import speckle_tracking as st
import h5py
import numpy as np


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

"""
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
mask = f[output_group+'mask'][()]
W    = f[output_group+'whitefield'][()]
roi  = f[output_group+'roi'][()]
defocus = f[output_group+'defocus'][()]
defocus_fs = f[output_group+'defocus_fs'][()]
defocus_ss = f[output_group+'defocus_ss'][()]
f.close()

dz = defocus - defocus_fs

pm0, pixel_map_inv, dxy = st.make_pixel_map(
                                   z, defocus, dz, roi,
                                   x_pixel_size, y_pixel_size,
                                   W.shape)

dij_n = st.make_pixel_translations(translations, basis, dxy[0], dxy[1])

O, n0, m0 = st.make_object_map(data, mask, W, dij_n, pm0, subpixel=False)

pixel_map, map_mask, res = st.update_pixel_map(data, mask, W, O, pm0, n0, m0, dij_n, 
                           roi=roi, guess=True)

O, n0, m0 = st.make_object_map(data, map_mask, W, dij_n, pixel_map, subpixel=True)

error_total, error_frame, error_pixel = st.calc_error(
                                        data, map_mask, W, dij_n, O, 
                                        pixel_map, n0, m0, subpixel=True)

write = {'object_map': O, 'n0': n0, 'm0': m0, 
         'pixel_map': pixel_map, 
         'map_mask': map_mask, 'error_pixel': error_pixel}

f = h5py.File('siemens_star.cxi')
for k in write.keys():
    key = output_group+k
    if key in f :
        del f[key]
    f[key] = write[k]
f.close()
