import numpy as np
import tqdm

def make_projection_images(mask, W, O, pixel_map, n0, m0, dij_n):
    out = -np.ones((len(dij_n),) + W.shape, dtype=np.float) 
    t   = np.zeros((np.sum(mask),), dtype=np.float)
    
    # mask the pixel mapping
    ij     = np.array([pixel_map[0][mask], pixel_map[1][mask]])
        
    for n in range(out.shape[0]):
        ss = np.rint(ij[0] - dij_n[n, 0] + n0).astype(np.int)
        fs = np.rint(ij[1] - dij_n[n, 1] + m0).astype(np.int)
        
        m2 = (ss>0)*(ss<O.shape[0])*(fs>0)*(fs<O.shape[1])
        t       = W[mask] 
        t[m2]  *= O[ss[m2], fs[m2]]
        t[~m2]  = -1
        
        out[n][mask] = t
    
    return out 

def update_pixel_map(data, mask, W, O, pixel_map, n0, m0, dij_n, search_window=3, quadratic_refinement=False, filter=1., verbose=True):
    r"""
    Update the pixel_map by minimising an error metric within the search_window.
    
    Parameters
    ----------
    data : ndarray, float32, (N, M, L)
        Input diffraction data :math:`I^{z_1}_\phi`, the :math:`^{z_1}` 
        indicates the distance between the virtual source of light and the 
        :math:`_\phi` indicates the phase of the wavefront incident on the 
        sample surface. The dimensions are given by:
        
        - N = number of frames
        - M = number of pixels along the slow scan axis of the detector
        - L = number of pixels along the fast scan axis of the detector
    
    mask : ndarray, bool, (M, L)
        Detector good / bad pixel mask :math:`M`, where True indicates a good
        pixel and False a bad pixel.
    
    W : ndarray, float, (M, L)
        The whitefield image :math:`W`. This is the image one obtains without a 
        sample in place.

    pixel_map : ndarray, float, (2, M, L)
        An array containing the pixel mapping 
        between a detector frame and the object :math:`ij_\text{map}`, such that: 
        
        .. math:: 
        
            I^{z_1}_{\phi}[n, i, j]
            = W[i, j] I^\infty[&\text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0,\\
                               &\text{ij}_\text{map}[1, i, j] - \Delta ij[n, 1] + m_0]
    
    n0 : float
        Slow scan offset to the pixel mapping such that:
            
        .. math::
            
            \text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0 \ge -0.5 
            \quad\text{for all } i,j
    
    m0 : float
        Fast scan offset to the pixel mapping such that:
            
        .. math::
            
            \text{ij}_\text{map}[1, i, j] - \Delta ij[n, 1] + m_0 \ge -0.5 
            \quad\text{for all } i,j
    
    dij_n : ndarray, float, (N, 2)
        An array containing the sample shifts for each detector image in pixel units
        :math:`\Delta ij_n`.

    search_window : int, len 2 sequence, optional
        The pixel mapping will be updated in a square area of side length "search_window". 
        If "search_window" is a length 2 sequence (e.g. [8,12]) then the search area will
        be rectangular with [ss_range, fs_range]. This value/s are in pixel units.
    
    quadratic_refinement : bool, optional
        If true then a 2D quadratic will be fit to the error metric around the local minima
        the location of the minima of this quadratic will be used to get sub-pixel precision
        for "pixel_map".
    
    filter : None or float, optional
        If float then apply a gaussian filter to the pixel_maps, ignoring masked pixels. 
        The "filter" is equal to the sigma of the Gaussian in pixel units.
    
    verbose : bool, optional
        print what I'm doing.
    
    Returns
    -------
    pixel_map : ndarray, float, (2, M, L)
        An array containing the updated pixel mapping.

    res : dictionary
        A dictionary containing diagnostic information:

            - error_map : ndarray, float, (M, L) 
                The minimum value of the error metric at each detector pixel
    Notes
    -----
    The following error metric is minimised with respect to :math:`\text{ij}_\text{map}[0, i, j]`:

    .. math:: 
    
        \begin{align}
        \varepsilon[i, j] = 
            \sum_n \bigg(I^{z_1}_{\phi}[n, i, j]
            - W[i, j] I^\infty[&\text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0,\\
                               &\text{ij}_\text{map}[1, i, j] - \Delta ij[n, 1] + m_0]\bigg)^2
        \end{align}
    """
    #if verbose : print('Updating the pixel mapping using the object map:\n')
    out, res  = update_pixel_map_opencl(data, mask, W, O, pixel_map, n0, m0, dij_n, search_window)
    
    if quadratic_refinement :
        out, res2 = quadratic_refinement_opencl(data, mask, W, O, out, n0, m0, dij_n)
        res.update(res2)
    
    if filter is not None :
        from scipy.ndimage.filters import gaussian_filter
        out[0] = gaussian_filter(mask * out[0], filter, mode = 'constant')
        out[1] = gaussian_filter(mask * out[1], filter, mode = 'constant')
        norm   = gaussian_filter(mask.astype(np.float), filter, mode = 'constant')
        norm[norm==0.] = 1.
        out = out / norm
    
    return out, res

def update_pixel_map_np(data, mask, W, O, pixel_map, n0, m0, dij_n, search_window=3, window=0):
    r"""
    Notes
    -----
    .. math:: 
    
        \varepsilon[i, j] = 
            \sum_n \bigg(I^{z_1}_{\phi}[n, i, j]
            - W[i, j] I^\infty[&\text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0,\\
                               &\text{ij}_\text{map}[1, i, j] - \Delta ij[n, 1] + m_0]\bigg)^2
    """
    from scipy.ndimage.filters import gaussian_filter
    shifts = np.arange(-(search_window-1)//2, (search_window+1)//2, 1) 
    ij_out = pixel_map.copy()
    errors   = np.zeros((len(shifts)**2,)+W.shape, dtype=np.float) 
    overlaps = np.zeros((len(shifts)**2,)+W.shape, dtype=np.uint16) 
    
    # mask the pixel mapping
    ij     = np.array([pixel_map[0][mask], pixel_map[1][mask]])
    
    index = 0
    for i in shifts:
        for j in shifts:
            forw = make_projection_images(mask, W, O, pixel_map, n0, m0, dij_n-np.array([i,j]))
            for n in range(data.shape[0]):
                m = forw[n]>0
                errors[  index][m] += (data[n][m] - forw[n][m])**2
                overlaps[index][m] += 1
            
            # apply the window averaging
            if window is not 0 :
                errors[index]   = gaussian_filter(errors[index], window, mode = 'constant')
                overlaps[index] = gaussian_filter(overlaps[index], window, mode = 'constant')
            
            index += 1
            print(i, j)
        
    m = (overlaps >= 1)
    errors[m] /= overlaps[m]
    errors[~m] = np.inf
    
    # choose the delta ij with the lowest error
    i, j = np.unravel_index(np.argmin(errors, axis=0), (len(shifts), len(shifts)))
    ij_out[0] += shifts[i]
    ij_out[1] += shifts[j]
    return ij_out, errors, overlaps

def update_pixel_map_opencl_old(data, mask, W, O, pixel_map, n0, m0, dij_n, search_window=3):
    from scipy.ndimage.filters import gaussian_filter
    
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
    update_pixel_map_cl = program.update_pixel_map
    
    update_pixel_map_cl.set_scalar_arg_dtypes(
                        8*[None] + 2*[np.float32] + 7*[np.int32])
    
    # Get the max work group size for the kernel test on our device
    max_comp = device.max_compute_units
    max_size = update_pixel_map_cl.get_work_group_info(
                       cl.kernel_work_group_info.WORK_GROUP_SIZE, device)
    print('maximum workgroup size:', max_size)
    print('maximum compute units :', max_comp)
    
    # allocate local memory and dtype conversion
    ############################################
    localmem = cl.LocalMemory(np.dtype(np.float32).itemsize * data.shape[0])
    
    # inputs:
    Win         = W.astype(np.float32)
    pixel_mapin = pixel_map.astype(np.float32)
    Oin         = O.astype(np.float32)
    dij_nin     = dij_n.astype(np.float32)
    maskin      = mask.astype(np.int32)
    
    if type(search_window) is int :
        s_ss = search_window
        s_fs = search_window
    else :
        s_ss, s_fs = search_window
    
    ss_min, ss_max = (-(s_ss-1)//2, (s_ss+1)//2) 
    fs_min, fs_max = (-(s_fs-1)//2, (s_fs+1)//2) 
    
    # outputs:
    err_map      = np.empty(W.shape, dtype=np.float32)
    err_min      = np.empty(W.shape, dtype=np.float32)
    err_min.fill(np.finfo(np.float32).max)
    pixel_shift  = np.zeros(pixel_map.shape, dtype=np.float32)
    
    import time
    d0 = time.time()
    
    for ss_shift in range(ss_min, ss_max, 1):
        for fs_shift in range(fs_min, fs_max, 1):
            err_map.fill(np.finfo(np.float32).max)
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
                  n0, m0, 
                  data.shape[0], data.shape[1], data.shape[2], 
                  O.shape[0], O.shape[1], ss_shift, fs_shift)
            queue.finish()
             
            # apply a window smoothing operation to the error map
            m = (err_map < np.finfo(np.float32).max)
            err_map = gaussian_filter(m * err_map, 4., mode = 'constant').astype(np.float32)
            norm    = gaussian_filter(m, 4., mode = 'constant').astype(np.float32)
            norm[norm==0.] = 1.
            err_map = err_map / norm
            
            # update good pixels where err_map < err_min
            m = m * (err_map<err_min)
            err_min[m] = err_map[m]
            pixel_shift[0][m] = ss_shift
            pixel_shift[1][m] = fs_shift
    
    out = pixel_map + pixel_shift
    pixel_mapin += pixel_shift
    
    # qudratic fit refinement
    pixel_shift.fill(0.)
    err_quad = np.empty((9,) + W.shape, dtype=np.float32)
    
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
                  n0, m0, 
                  data.shape[0], data.shape[1], data.shape[2], 
                  O.shape[0], O.shape[1], ss_shift, fs_shift)
            queue.finish()
             
            # apply a window smoothing operation to the error map
            m = (err_map < np.finfo(np.float32).max)
            err_map = gaussian_filter(m * err_map, 4., mode = 'constant').astype(np.float32)
            norm    = gaussian_filter(m, 4., mode = 'constant').astype(np.float32)
            norm[norm==0.] = 1.
            err_map = err_map / norm
            
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
    
    #out[0][m] = out[0][m] + pixel_shift[0][m]
    #out[1][m] = out[1][m] + pixel_shift[1][m]
      
    print('calculation took:', time.time()-d0, 's')
    
    return out, {'error_map': mask*err_min, 'pixel_shift': pixel_shift, 'err_quad': err_quad}


def update_pixel_map_opencl(data, mask, W, O, pixel_map, n0, m0, dij_n, search_window=20):
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
    update_pixel_map_cl = program.update_pixel_map_old_subpixel
    
    update_pixel_map_cl.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None,
             np.float32, np.float32, np.int32, np.int32, 
             np.int32, np.int32, np.int32, np.int32, np.int32,
             np.int32, np.int32, np.int32])
    
    # Get the max work group size for the kernel test on our device
    max_comp = device.max_compute_units
    max_size = update_pixel_map_cl.get_work_group_info(
                       cl.kernel_work_group_info.WORK_GROUP_SIZE, device)
    #print('maximum workgroup size:', max_size)
    #print('maximum compute units :', max_comp)
    
    if type(search_window) is int :
        s_ss = search_window
        s_fs = search_window
    else :
        s_ss, s_fs = search_window
    
    # allocate local memory and dtype conversion
    ############################################
    localmem = cl.LocalMemory(np.dtype(np.float32).itemsize * data.shape[0])
    
    # inputs:
    Win         = W.astype(np.float32)
    pixel_mapin = pixel_map.astype(np.float32)
    Oin         = O.astype(np.float32)
    dij_nin     = dij_n.astype(np.float32)
    maskin      = mask.astype(np.int32)
    
    ss_min, ss_max = (-(s_ss-1)//2, (s_ss+1)//2) 
    fs_min, fs_max = (-(s_fs-1)//2, (s_fs+1)//2) 

    # outputs:
    err_map      = np.zeros(W.shape, dtype=np.float32)
    pixel_mapout = pixel_map.astype(np.float32)
    
    for i in tqdm.trange(W.shape[0], desc='updating pixel map'):
        update_pixel_map_cl(queue, (1, W.shape[1]), (1, 1), 
              cl.SVM(Win), 
              cl.SVM(data), 
              localmem, 
              cl.SVM(err_map), 
              cl.SVM(Oin), 
              cl.SVM(pixel_mapout), 
              cl.SVM(dij_nin), 
              cl.SVM(maskin),
              n0, m0, 
              data.shape[0], data.shape[1], data.shape[2], 
              O.shape[0], O.shape[1], ss_min, ss_max, fs_min, fs_max, i)
     
        queue.finish()
    
    return pixel_mapout, {'error_map': err_map}


def quadratic_refinement_opencl(data, mask, W, O, pixel_map, n0, m0, dij_n):
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
    update_pixel_map_cl = program.update_pixel_map
    
    update_pixel_map_cl.set_scalar_arg_dtypes(
                        8*[None] + 2*[np.float32] + 7*[np.int32])
    
    # Get the max work group size for the kernel test on our device
    max_comp = device.max_compute_units
    max_size = update_pixel_map_cl.get_work_group_info(
                       cl.kernel_work_group_info.WORK_GROUP_SIZE, device)
    print('maximum workgroup size:', max_size)
    print('maximum compute units :', max_comp)
    
    # allocate local memory and dtype conversion
    ############################################
    localmem = cl.LocalMemory(np.dtype(np.float32).itemsize * data.shape[0])
    
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
    
    return out, {'pixel_shift': pixel_shift, 'err_quad': err_quad}
