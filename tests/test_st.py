import h5py
import numpy as np
import speckle_tracking as st

def main(path, roi, df, ls, n_iter):
    with h5py.File(path, 'r') as cxi_file:
        I_n = cxi_file['/entry_1/data_1/data'][:, roi[0]:roi[1], roi[2]:roi[3]].sum(axis=1)[:, None]
        basis = cxi_file['/entry_1/instrument_1/detector_1/basis_vectors'][...]
        z = cxi_file['/entry_1/instrument_1/detector_1/distance'][...]
        x_ps = cxi_file['/entry_1/instrument_1/detector_1/x_pixel_size'][...]
        y_ps = cxi_file['/entry_1/instrument_1/detector_1/y_pixel_size'][...]
        wl = cxi_file['/entry_1/instrument_1/source_1/wavelength'][...]
        dij = cxi_file['/entry_1/sample_1/geometry/translation'][...]
    M = np.ones((I_n.shape[1], I_n.shape[2]), dtype=bool)
    W = st.make_whitefield(I_n, M).astype(I_n.dtype)
    u, dij_pix, res = st.generate_pixel_map(W.shape, dij, basis, x_ps,
                                            y_ps, z, df)
    print(I_n.dtype, W.dtype, dij_pix.dtype, u.dtype)
    I0, n0, m0 = st.make_object_map(data=I_n, mask=M, W=W, dij_n=dij_pix, pixel_map=u, ls=ls)

    es = []
    for i in range(n_iter):

        # calculate errors
        error_total = st.calc_error(data=I_n, mask=M, W=W, dij_n=dij_pix, O=I0, pixel_map=u, n0=n0, m0=m0, ls=ls, subpixel=False, verbose=False)[0]

        # store total error
        es.append(error_total)
        # update pixel map
        u = st.update_pixel_map(data=I_n, mask=M, W=W, O=I0, pixel_map=u, n0=n0, m0=m0, dij_n=dij_pix,
                                sw_ss=0, sw_fs=10, ls=ls)

        # make reference image
        I0, n0, m0 = st.make_object_map(data=I_n, mask=M, W=W, dij_n=dij_pix, pixel_map=u, ls=ls)

if __name__  == '__main__':
    main('../pyrost/results/exp/Scan_2008.cxi', (0, 1, 350, 1200), 1e-4, 0.1, 5)
    print('Test is done.')
