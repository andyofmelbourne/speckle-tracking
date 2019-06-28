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


#---------------------------
# intialise
#---------------------------
mask  = st.make_mask(data)

W = st.make_whitefield(data, mask)

roi = st.guess_roi(W)

defocus, res = st.fit_defocus(
             data,
             x_pixel_size, y_pixel_size,
             z, wav, mask, W, roi)

#pixel_map, uinv, dxy = st.make_pixel_map(
#                  z, defocus, res['astigmatism'], 
#                  roi, x_pixel_size, y_pixel_size,
#                  W.shape)

M = z / defocus 

dx = dy = x_pixel_size / M

xy = st.make_pixel_translations(
                translations, 
                basis, dx, dy)

u, res = st.pixel_map_from_data(data, xy[:, 0], xy[:, 1], W, mask)

#O, n0, m0 = st.make_object_map(
#               data, mask, W, dij_n, pixel_map)

"""
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


phase, angles, res = st.integrate_pixel_map(pixel_map, W, wav, z-z1, z, x_pixel_size, y_pixel_size, dxy[0], dxy[1], False, maxiter=5000)

propx, propy, dx, dy, dz = st.propagation_profile(phase, W, z, wav, x_pixel_size, y_pixel_size, 1, zs=[-1e-4, 1e-4, 1000], Nint=4)

#---------------------------
# Write
#---------------------------
st.write_h5({
    'object_map': O, 
    'object_map_voxel_size': dxy, 
    'n0': n0, 'm0': m0, 
    'pixel_map': pixel_map, 
    'pixel_translations': dij_n,
    'propagation_profile_ss': propx, 
    'propagation_profile_fs': propy, 
    'propagation_profile_voxel_size': np.array([dx, dy, dz]),
    'phase' : phase,
    'angles' : angles,
    'angles_forward' : res['angles_forward']
    })
"""
