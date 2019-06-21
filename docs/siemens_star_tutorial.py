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


