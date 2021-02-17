import h5py
import numpy as np
import speckle_tracking as st
import pytest

@pytest.fixture(params=[1., 15.])
def length_scale(request):
    return request.param

@pytest.mark.st
def test_st_update(defocus, length_scale, path, roi):
    with h5py.File(path, 'r') as cxi_file:
            I_n = cxi_file['/entry_1/data_1/data'][:, roi[0]:roi[1], roi[2]:roi[3]].sum(axis=1)[:, None]
            basis = cxi_file['/entry_1/instrument_1/detector_1/basis_vectors'][...]
            z = cxi_file['/entry_1/instrument_1/detector_1/distance'][...]
            x_ps = cxi_file['/entry_1/instrument_1/detector_1/x_pixel_size'][...]
            y_ps = cxi_file['/entry_1/instrument_1/detector_1/y_pixel_size'][...]
            wl = cxi_file['/entry_1/instrument_1/source_1/wavelength'][...]
            dij = cxi_file['/entry_1/sample_1/geometry/translation'][...]
    M = np.ones((I_n.shape[1], I_n.shape[2]), dtype=bool)
    W = st.make_whitefield(I_n, M, verbose=True).astype(I_n.dtype)
    u, dij_pix, res = st.generate_pixel_map(W.shape, dij, basis, x_ps,
                                            y_ps, z, defocus, verbose=True)
    I0, n0, m0 = st.make_object_map(data=I_n, mask=M, W=W, dij_n=dij_pix, pixel_map=u, ls=length_scale)

    es = []
    for i in range(5):

        # calculate errors
        error_total = st.calc_error(data=I_n, mask=M, W=W, dij_n=dij_pix, O=I0, pixel_map=u, n0=n0, m0=m0,
                                    ls=length_scale, verbose=True)[0]

        # store total error
        es.append(error_total)
        # update pixel map
        u = st.update_pixel_map(data=I_n, mask=M, W=W, O=I0, pixel_map=u, n0=n0, m0=m0, dij_n=dij_pix,
                                sw_ss=0, sw_fs=10, ls=length_scale)

        # make reference image
        I0, n0, m0 = st.make_object_map(data=I_n, mask=M, W=W, dij_n=dij_pix, pixel_map=u, ls=length_scale)
