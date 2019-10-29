import h5py
import numpy as np

f = h5py.File('hdf5/diatom/MLL_260.cxi')
pix = f['build_atlas/pix_positions_dist'][()]
pixout = np.zeros((121, 3), dtype=np.float)
pixout[1:] = pix
del f['build_atlas/pix_positions_dist']
f['build_atlas/pix_positions_dist'] = pixout
f.close()
