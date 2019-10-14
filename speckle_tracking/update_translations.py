import numpy as np
import tqdm
from .make_object_map import make_object_map

def update_translations(
        data, mask, W, O, pixel_map, 
        n0, m0, dij_n, search_window=[3,3], 
        update_O=False, maxiters=100, tol=1e-3, 
        roi=None):
    
    if roi is None :
        roi = [0, W.shape[0], 0, W.shape[1]]
    
    dr = data[:, roi[0]:roi[1], roi[2]:roi[3]].copy()
    mr = mask[roi[0]:roi[1], roi[2]:roi[3]].copy()
    Wr = W[roi[0]:roi[1], roi[2]:roi[3]].copy()
    ur = pixel_map[:, roi[0]:roi[1], roi[2]:roi[3]].copy()

    out = dij_n.copy()
    # first update O and xy until convergence
    it = tqdm.trange(maxiters, desc='updating object map and translations')
    for i in it:
        O, n0, m0 = make_object_map(dr, mr, Wr, out, ur, subpixel=False)
        out, res  = update_translations_1(dr, mr, Wr, O, ur, n0, m0, out)
        err_new   = res['error']
        
        if i>0 and (err_old - err_new)/err_old < tol :
            it.close()
            break
        err_old = err_new
        it.set_description("updating object map and translations: {:.2e}".format(res['error']))
    return out, res

def update_translations_1(data, mask, W, O, pixel_map, n0, m0, dij_n, search_window=[3,3]):
    N = dij_n.shape[0]
    # add 2 to the search window so that we can do quadratic refinement
    s2 = np.array(search_window)+2
    
    ss_min, ss_max = (-(s2[0]-1)//2, (s2[0]+1)//2) 
    fs_min, fs_max = (-(s2[1]-1)//2, (s2[1]+1)//2) 
    
    ss = np.arange(ss_min, ss_max, 1)
    fs = np.arange(fs_min, fs_max, 1)
    ss, fs = np.meshgrid(ss, fs, indexing='ij')
    ss, fs = ss.ravel(), fs.ravel()
    
    errs = calc_errs(data, mask, W, O, pixel_map, n0, m0, dij_n, ss, fs)
    err1 = np.sum(np.min(errs, axis=0))
    errs = errs.reshape((s2[0], s2[1], N))
    t    = errs[1:-1, 1:-1, :].reshape((search_window[0] * search_window[1], N))
    err0 = np.sum(errs[s2[0]//2, s2[1]//2, :])
    
    errs_quad = np.zeros((3*3, N), dtype=np.float)
    out       = dij_n.copy()
    # now define the errors around the minimum 
    # in a 3x3 window for quadratic refinement

    # there are two coordinates here
    # k : i, j -> search window
    # kk: u, v -> search window + 2
    for n in range(N):
        k      = np.argmin(t[:, n].ravel())
        i, j   = np.unravel_index(k, tuple(search_window))
        u, v   = i+1, j+1
        kk     = u*s2[1] + v
        out[n, 0] += ss[kk]
        out[n, 1] += fs[kk]
        l = 0
        for ii in [-1, 0, 1]:
            for jj in [-1, 0, 1]:
                errs_quad[l, n] = errs[u+ii, v+jj, n]
                l += 1
    
    A = []
    #print('\nquadratic refinement:')
    #print('---------------------')
    for ss_shift in [-1, 0, 1]:
        for fs_shift in [-1, 0, 1]:
            A.append([ss_shift**2, fs_shift**2, ss_shift, fs_shift, ss_shift*fs_shift, 1])
    
    # now we have 9 equations and 6 unknowns
    # c_20 x^2 + c_02 y^2 + c_10 x + c_01 y + c_11 x y + c_00 = err_i
    B = np.linalg.pinv(A)
    C = np.dot(B, errs_quad) # C.shape = (6, N)
    
    # minima is defined by
    # 2 c_20 x +   c_11 y = -c_10
    #   c_11 x + 2 c_02 y = -c_01
    # where C = [c_20, c_02, c_10, c_01, c_11, c_00]
    #           [   0,    1,    2,    3,    4,    5]
    # [x y] = [[2c_02 -c_11], [-c_11, 2c_20]] . [-c_10 -c_01] / (2c_20 * 2c_02 - c_11**2)
    # x     = (-2c_02 c_10 + c_11   c_01) / det
    # y     = (  c_11 c_10 - 2 c_20 c_01) / det
    det  = 2*C[0] * 2*C[1] - C[4]**2
    
    # make sure all sampled shifts have a valid error
    # make sure the determinant is non zero
    m    = (det != 0)
    det[~m]  = 1
    ss_shift = (-2*C[1] * C[2] +   C[4] * C[3]) / det
    fs_shift = (   C[4] * C[2] - 2*C[0] * C[3]) / det
    
    A = np.array([ss_shift**2, fs_shift**2, ss_shift, fs_shift, ss_shift*fs_shift, np.ones_like(ss_shift)])
    err2 = np.sum( np.dot(C.T, A) )
    
    #print('input error: {:.3e}'.format(err0))
    #print('min   error: {:.3e}'.format(err1))
    #print('quad  error: {:.3e}'.format(err2))
    
    m *= (ss_shift**2+fs_shift**2 < 9)
    out[m, 0] = ss_shift[m] + out[m, 0]
    out[m, 1] = fs_shift[m] + out[m, 1]
    return out, {'error': err1, 'errs_quad': errs_quad}

def calc_errs(data, mask, W, O, pixel_map, n0, m0, dij_n, ss, fs):
    # demand that the data is float32 to avoid excess mem. usage
    assert(data.dtype == np.float32)
    assert(ss.dtype == np.int)
    assert(fs.dtype == np.int)
    
    import os
    import pyopencl as cl
    ## Step #1. Obtain an OpenCL platform.
    # with a cpu device
    for p in cl.get_platforms():
        devices = p.get_devices(cl.device_type.CPU)
        if len(devices) > 0:
            platform = p
            device   = devices[0]
            break
    
    ## Step #3. Create a context for the selected device.
    context = cl.Context([device])
    queue   = cl.CommandQueue(context)
    
    # load and compile the update_pixel_map opencl code
    here = os.path.split(os.path.abspath(__file__))[0]
    kernelsource = os.path.join(here, 'update_pixel_map.cl')
    kernelsource = open(kernelsource).read()
    program     = cl.Program(context, kernelsource).build()
    translations_err_cl = program.translations_err
    
    translations_err_cl.set_scalar_arg_dtypes(
                        8*[None] + 2*[np.float32] + 6*[np.int32])
    
    # Get the max work group size for the kernel test on our device
    max_comp = device.max_compute_units
    max_size = translations_err_cl.get_work_group_info(
                       cl.kernel_work_group_info.WORK_GROUP_SIZE, device)
    #print('maximum workgroup size:', max_size)
    #print('maximum compute units :', max_comp)
    
    # allocate local memory and dtype conversion
    ############################################
    localmem = cl.LocalMemory(np.dtype(np.float32).itemsize * data.shape[0])
    
    # inputs:
    Win         = W.astype(np.float32)
    pixel_mapin = pixel_map.astype(np.float32)
    Oin         = O.astype(np.float32)
    dij_nin     = dij_n.astype(np.float32)
    maskin      = mask.astype(np.int32)
    ns          = np.arange(data.shape[0]).astype(np.int32)
    
    # outputs:
    dij_nout     = dij_n.copy()
    errs         = np.empty((len(ss), data.shape[0]), dtype=np.float32)
    out          = np.zeros(data.shape[0]).astype(np.float32)
    
    step = max_comp
    for i in range(len(ss)):
        for n in tqdm.tqdm(np.arange(ns.shape[0])[::step], desc='updating sample translations'):
            nsi = ns[n:n+step:]
            translations_err_cl( queue, (nsi.shape[0], 1), (1, 1), 
                  cl.SVM(Win), 
                  cl.SVM(data), 
                  cl.SVM(Oin), 
                  cl.SVM(pixel_mapin), 
                  cl.SVM(dij_nin), 
                  cl.SVM(maskin),
                  cl.SVM(nsi),
                  cl.SVM(out),
                  n0, m0, 
                  data.shape[1], data.shape[2], 
                  O.shape[0], O.shape[1], ss[i], fs[i])
            queue.finish()
                
            errs[i] = out
    
    return errs
