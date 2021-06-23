# update the pixel_map on each split 1/2 of the data
# keeping the object map and translations constant

from .update_pixel_map import *

def angular_resolution(data, mask, W, O, pixel_map, n0, m0, dij_n, dss, dfs, z,
                     search_window=None, grid=None, roi=None, 
                     subpixel=False, subsample=1., 
                     interpolate = False, fill_bad_pix=True,
                     quadratic_refinement = True,
                     integrate = False, clip = None, 
                     filter=None, verbose=False, guess=False):
    """
    """
    
    split_mask = np.random.random(data.shape)>= 0.5 
    
    u1, res1 = update_pixel_map_split(
                     data, mask, split_mask, W, O, pixel_map, n0, m0, dij_n, 
                     search_window, grid, roi, 
                     subpixel, subsample, 
                     interpolate, fill_bad_pix,
                     quadratic_refinement,
                     integrate, clip, 
                     filter, verbose, guess)

    u2, res2 = update_pixel_map_split(
                     data, mask, ~split_mask, W, O, pixel_map, n0, m0, dij_n, 
                     search_window, grid, roi, 
                     subpixel, subsample, 
                     interpolate, fill_bad_pix,
                     quadratic_refinement,
                     integrate, clip, 
                     filter, verbose, guess)
    
    # convert to angles 
    dt    = u2-u1
    x1, y1, g1, s1 = fit_gauss(dt[0])
    x2, y2, g2, s2 = fit_gauss(dt[1])
    s1 *= dss / z
    s2 *= dfs / z
    x1 *= dss / z
    x2 *= dfs / z
    
    res3 = {'x_ss': x1, 'h_ss': y1, 'h_fit_ss': g1, 'sigma_ss': s1,
            'x_fs': x2, 'h_fs': y2, 'h_fit_fs': g2, 'sigma_fs': s2}
    
    return u1, u2, res1, res2, res3

def fit_gauss(u):
    # now fit a gaussian to the histogram 
    hist, bins = np.histogram(u, bins=np.arange(u.min(), u.max(), 0.01))
    
    x = bins[:-1] 
    y = hist
    from scipy.optimize import curve_fit
    # weighted arithmetic mean (corrected - check the section below)
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

    def Gauss(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    popt, pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])
    return x, y, Gauss(x, *popt), popt[2]

def update_pixel_map_split(data, mask, split_mask, W, O, pixel_map, n0, m0, dij_n, 
                     search_window=None, grid=None, roi=None, 
                     subpixel=False, subsample=1., 
                     interpolate = False, fill_bad_pix=True,
                     quadratic_refinement = True,
                     integrate = False, clip = None, 
                     filter=None, verbose=False, guess=False):
    # any parameter that the user specifies should be enforced
    # We should have "None" mean: please guess it for me
    if roi is None :
        roi = [0, W.shape[0], 0, W.shape[1]]
    
    # define search_window 
    if search_window is None :
        from .calc_error import make_pixel_map_err
        ijs, err_map, res = make_pixel_map_err(
                            data, mask, W, O, pixel_map, n0, m0, 
                            dij_n, roi, search_window=100, grid=[10, 10])
        
        search_window = res['search_window']
    
    elif type(search_window) is int :
        search_window = [search_window, search_window]
    
    # define grid
    if grid is None :
        grid = [roi[1]-roi[0], roi[3]-roi[2]]
    
    ss_grid = np.linspace(roi[0], roi[1]-1, grid[0])
    fs_grid = np.linspace(roi[2], roi[3]-1, grid[1])
    ss_grid, fs_grid = np.meshgrid(ss_grid, fs_grid, indexing='ij')
    
    # grid search of pixel shifts
    u, res = update_pixel_map_split_opencl(
               data, split_mask * mask, W, O, pixel_map, n0, m0, 
               dij_n, subpixel, subsample, 
               search_window, ss_grid.ravel(), fs_grid.ravel())
    
    # if the update is on a sparse grid, then interpolate
    if interpolate :
        out, map_mask = interpolate_pixel_map(
                        out.reshape((2,) + ss_grid.shape), ss_grid, fs_grid, mask, grid, roi)
    else :
        out = pixel_map.copy()
        ss, fs = np.rint(ss_grid).astype(np.int), np.rint(fs_grid).astype(np.int)
        out[0][ss, fs] = u[0].reshape(ss_grid.shape)
        out[1][ss, fs] = u[1].reshape(ss_grid.shape)

    if quadratic_refinement :
        out, res = quadratic_refinement_split_opencl(data, mask*split_mask, W, O, out, n0, m0, dij_n)

    if fill_bad_pix :
        out[0] = fill_bad(out[0], mask, 4.)
        out[1] = fill_bad(out[1], mask, 4.)

    if integrate :
        from .utils import integrate
        u0 = np.array(np.indices(W.shape))
        phase_pix, res = integrate(
                         out[0]-u0[0], out[1]-u0[1], 
                         W**0.5, maxiter=2000)
        
        if clip is not None :
            out[0] = u0[0] + np.clip(res['dss_forward'], clip[0], clip[1])
            out[1] = u0[1] + np.clip(res['dfs_forward'], clip[0], clip[1])
        else :
            out[0] = u0[0] + res['dss_forward']
            out[1] = u0[1] + res['dfs_forward']
    
    if (filter is not None) and (filter > 0):
        out = filter_pixel_map(out, mask, filter)
    return out, res

def update_pixel_map_split_opencl(
        data, mask, W, O, pixel_map, n0, m0, 
        dij_n, subpixel, subsample, search_window, 
        ss, fs):
    # demand that the data is float32 to avoid excess mem. usage
    assert(data.dtype == np.float32)
    
    ##################################################################
    # OpenCL crap
    ##################################################################
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
    
    update_pixel_map_cl = program.update_pixel_map_split
    
    update_pixel_map_cl.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None, None, None, None,
             np.float32, np.float32, np.float32, np.int32, np.int32, 
             np.int32, np.int32, np.int32, np.int32, np.int32,
             np.int32, np.int32])
    
    # Get the max work group size for the kernel test on our device
    max_comp = device.max_compute_units
    max_size = update_pixel_map_cl.get_work_group_info(
                       cl.kernel_work_group_info.WORK_GROUP_SIZE, device)
    #print('maximum workgroup size:', max_size)
    #print('maximum compute units :', max_comp)
    
    # allocate local memory and dtype conversion
    ############################################
    localmem = cl.LocalMemory(np.dtype(np.float32).itemsize * data.shape[0])
    localmem_mask = cl.LocalMemory(np.dtype(np.int32).itemsize * mask.shape[0])
    
    # inputs:
    Win         = W.astype(np.float32)
    pixel_mapin = pixel_map.astype(np.float32)
    Oin         = O.astype(np.float32)
    dij_nin     = dij_n.astype(np.float32)
    maskin      = mask.astype(np.int32)
    ss          = ss.ravel().astype(np.int32)
    fs          = fs.ravel().astype(np.int32)
    
    ss_min, ss_max = (-(search_window[0]-1)//2, (search_window[0]+1)//2) 
    fs_min, fs_max = (-(search_window[1]-1)//2, (search_window[1]+1)//2) 
    
    # outputs:
    err_map      = np.zeros(W.shape, dtype=np.float32)
    pixel_mapout = pixel_map.astype(np.float32)
    ##################################################################
    # End crap
    ##################################################################
    
    step = min(100, ss.shape[0])
    for i in tqdm.tqdm(np.arange(ss.shape[0])[::step], desc='updating pixel map'):
        ssi = ss[i:i+step:]
        fsi = fs[i:i+step:]
        update_pixel_map_cl(queue, (1, fsi.shape[0]), (1, 1), 
              cl.SVM(Win), 
              cl.SVM(data), 
              localmem, 
              cl.SVM(err_map), 
              cl.SVM(Oin), 
              cl.SVM(pixel_mapout), 
              cl.SVM(dij_nin), 
              cl.SVM(maskin),
              localmem_mask, 
              cl.SVM(ssi),
              cl.SVM(fsi),
              n0, m0, subsample,
              data.shape[0], data.shape[1], data.shape[2], 
              O.shape[0], O.shape[1], ss_min, ss_max, fs_min, fs_max)
        queue.finish()
    
    # only return filled values
    out = np.zeros((2,) + ss.shape, dtype=pixel_map.dtype)
    out[0] = pixel_mapout[0][ss, fs]
    out[1] = pixel_mapout[1][ss, fs]
    return out, {'error_map': err_map}

def quadratic_refinement_split_opencl(data, mask, W, O, pixel_map, n0, m0, dij_n):
    # demand that the data is float32 to avoid excess mem. usage
    assert(data.dtype == np.float32)
    
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
    update_pixel_map_cl = program.pixel_map_err_split
    
    update_pixel_map_cl.set_scalar_arg_dtypes(
                        9*[None] + 2*[np.float32] + 7*[np.int32])
    
    # Get the max work group size for the kernel test on our device
    max_comp = device.max_compute_units
    max_size = update_pixel_map_cl.get_work_group_info(
                       cl.kernel_work_group_info.WORK_GROUP_SIZE, device)
    print('maximum workgroup size:', max_size)
    print('maximum compute units :', max_comp)
    
    # allocate local memory and dtype conversion
    ############################################
    localmem = cl.LocalMemory(np.dtype(np.float32).itemsize * data.shape[0])
    localmem_mask = cl.LocalMemory(np.dtype(np.int32).itemsize * mask.shape[0])
    
    # inputs:
    Win         = W.astype(np.float32)
    pixel_mapin = pixel_map.astype(np.float32)
    Oin         = O.astype(np.float32)
    dij_nin     = dij_n.astype(np.float32)
    maskin      = mask.astype(np.int32)
    
    # outputs:
    err_map      = np.empty(W.shape, dtype=np.float32)
    pixel_shift  = np.zeros(pixel_map.shape, dtype=np.float32)
    err_quad     = np.empty((9,) + W.shape, dtype=np.float32)
    out          = pixel_map.copy()
    
    import time
    d0 = time.time()
    
    # qudratic fit refinement
    pixel_shift.fill(0.)
    
    A = []
    print('\nquadratic refinement:')
    print('---------------------')
    for ss_shift in [-1, 0, 1]:
        for fs_shift in [-1, 0, 1]:
            A.append([ss_shift**2, fs_shift**2, ss_shift, fs_shift, ss_shift*fs_shift, 1])
            print(ss_shift, fs_shift)
            update_pixel_map_cl( queue, W.shape, (1, 1), 
                  cl.SVM(Win), 
                  cl.SVM(data), 
                  localmem, 
                  cl.SVM(err_map), 
                  cl.SVM(Oin), 
                  cl.SVM(pixel_mapin), 
                  cl.SVM(dij_nin), 
                  cl.SVM(maskin),
                  localmem_mask, 
                  n0, m0, 
                  data.shape[0], data.shape[1], data.shape[2], 
                  O.shape[0], O.shape[1], ss_shift, fs_shift)
            queue.finish()
            
            err_quad[3*(ss_shift+1) + fs_shift+1, :, :] = err_map
    
    # now we have 9 equations and 6 unknowns
    # c_20 x^2 + c_02 y^2 + c_10 x + c_01 y + c_11 x y + c_00 = err_i
    B = np.linalg.pinv(A)
    C = np.dot(B, np.transpose(err_quad, (1, 0, 2)))
    
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
    m    = np.all(err_quad<np.finfo(np.float32).max, axis=0)
    # make sure the determinant is non zero
    m    = m * (det != 0)
    pixel_shift[0][m] = (-2*C[1] * C[2] +   C[4] * C[3])[m] / det[m]
    pixel_shift[1][m] = (   C[4] * C[2] - 2*C[0] * C[3])[m] / det[m]
    
    # now only update pixels for which (x**2 + y**2) < 3**2
    m = m * (np.sum(pixel_shift**2, axis=0) < 9)
    
    out[0][m] = out[0][m] + pixel_shift[0][m]
    out[1][m] = out[1][m] + pixel_shift[1][m]
      
    print('calculation took:', time.time()-d0, 's')
    
    error = np.sum(np.min(err_quad, axis=0))
    return out, {'pixel_shift': pixel_shift, 'error': error, 'err_quad': err_quad}
