import numpy as np
import tqdm

from .make_object_map import make_object_map

def calc_error(data, mask, W, dij_n, I, pixel_map, n0, m0, subpixel=False, verbose=True):
    r"""
    Parameters
    ----------
    data : ndarray
        Input data, of shape (N, M, L).
    
    mask : ndarray
        Boolean array of shape (M, L), where True indicates a good
        pixel and False a bad pixel.
    
    W : ndarray
        Float array of shape (M, L) containing the estimated whitefield.
    
    dij_n : ndarray
        Float array of shape (N, 2) containing the object translations 
        that have been mapped onto the detector's frame of reference.     

    I : ndarray
        Float array of shape (U, V), this is essentially an object map. 
    
    pixel_map : ndarray, (2, M, L)
        An array containing the pixel mapping 
        between a detector frame and the object, such that: 
        
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
    
    subpixel : bool, optional
        If True then use bilinear subpixel interpolation non-integer pixel mappings.
    
    verbose : bool, optional
        print what I'm doing.
    
    minimum_overlap : float or None, optional
        Default is None. If float then the the object will be set to -1 
        where the number of data points contributing to that value is less
        than "minimum_overlap".

    Returns
    -------
    
    error_total : float
        The global error value, :math:`\varepsilon = \sum_{n,i,j} \varepsilon[n, i, j]`.
    
    error_frame : ndarray
        Float array of shape (N,). The average pixel error per detector frame, 
        :math:`\varepsilon_\text{frame}[n] = \langle \varepsilon[n, i, j] \rangle_{i,j}`.
        
    error_pixel : ndarray
        Float array of shape (M, L). The average pixel error per detector pixel, 
        :math:`\varepsilon_\text{pixel}[i, j] = \langle \varepsilon[n, i, j]\rangle_{n}`.
    
    Notes
    -----
    The error, per pixel and per frame, is given by:

    .. math::

        \begin{align}
        \varepsilon[n, i, j] = M[i,j] \bigg[ I_\Phi[n, i, j] - 
            W[i, j] I_0[&\text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0,\\
                        &\text{ij}_\text{map}[1, i, j] - \Delta ij[n, 1] + m_0]\bigg]^2
        \end{align}
    """
    # mask the pixel mapping
    ij     = np.array([pixel_map[0], pixel_map[1]])
    
    error_total = 0.
    error_residual = np.zeros(data.shape, dtype=np.float32)
    error_frame = np.zeros(data.shape[0])
    error_pixel = np.zeros(data.shape[1:])
    norm        = np.zeros(data.shape[1:])
    flux_corr   = np.zeros(data.shape[0])
    
    #sig = np.std(data, axis=0)
    #sig[sig <= 0] = 1
    
    for n in tqdm.trange(data.shape[0], desc='calculating errors'):
        if subpixel: 
            # define the coordinate mapping and round to int
            ss = pixel_map[0] - dij_n[n, 0] + n0
            fs = pixel_map[1] - dij_n[n, 1] + m0
            #
            I0 = W * bilinear_interpolation_array(I, ss, fs, fill=-1, invalid=-1)
            #I0 = I0[mask]
        
        else :
            # define the coordinate mapping and round to int
            ss = np.rint((ij[0] - dij_n[n, 0] + n0)).astype(np.int)
            fs = np.rint((ij[1] - dij_n[n, 1] + m0)).astype(np.int)
            #
            #I0 = I[ss, fs] * W[mask]
            I0 = I[ss, fs] * W
        

        d  = data[n]
        m  = (I0>0)*(d>0)*mask
                
        # flux correction factor
        flux_corr[n] = np.sum(m * I0 * data[n]) / np.sum(m * data[n]**2) 
        
        #error_map = m*(I0 - d)**2 / sig[mask]
        error_map = m*(I0 - d)**2
        tot       = np.sum(error_map)
        
        error_total       += tot
        error_pixel       += error_map
        error_frame[n]     = tot / np.sum(m)
        #print(m.shape, mask.shape, mask[m].shape, W.shape, norm.shape)
        norm              += m*(W - d)**2
        error_map          = m * np.abs(W-d)
        error_map[~m]      = -1      
        error_residual[n]  = error_map
    
    # now map the errors to object space
    error_reference, n0, m0 = make_object_map(error_residual, mask, W, dij_n, pixel_map, subpixel=True)

    print("err ref shape:", error_residual.shape)
    
    m = norm>0
    error_pixel[m] = error_pixel[m] / norm[m]
    return error_total, error_frame, error_pixel, error_residual, error_reference, norm, flux_corr

def make_pixel_map_err(data, mask, W, O, pixel_map, n0, m0, dij_n, roi, search_window=20, grid=[20, 20]):
    # demand that the data is float32 to avoid excess mem. usage
    assert(data.dtype == np.float32)
    
    import time
    t0 = time.time()
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
    make_error_map_subpixel = program.make_error_map_subpixel

    make_error_map_subpixel.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None, None,
             np.float32, np.float32, 
             np.int32, np.int32, np.int32, np.int32, 
             np.int32, np.int32, np.int32, np.int32, 
             np.int32, np.int32])
    
    # Get the max work group size for the kernel test on our device
    max_comp = device.max_compute_units
    max_size = make_error_map_subpixel.get_work_group_info(
                       cl.kernel_work_group_info.WORK_GROUP_SIZE, device)
    #print('maximum workgroup size:', max_size)
    #print('maximum compute units :', max_comp)
    
    # allocate local memory and dtype conversion
    localmem = cl.LocalMemory(np.dtype(np.float32).itemsize * data.shape[0])
    
    # inputs:
    Win         = W.astype(np.float32)
    pixel_mapin = pixel_map.astype(np.float32)
    Oin         = O.astype(np.float32)
    dij_nin     = dij_n.astype(np.float32)
    maskin      = mask.astype(np.int32)
    
    # outputs:
    err_map      = np.zeros((grid[0]*grid[1], search_window**2), dtype=np.float32)
    pixel_mapout = pixel_map.astype(np.float32)
    ##################################################################
    
    if type(search_window) is int :
        s_ss = search_window
        s_fs = search_window
    else :
        s_ss, s_fs = search_window
    
    ss_min, ss_max = (-(s_ss-1)//2, (s_ss+1)//2) 
    fs_min, fs_max = (-(s_fs-1)//2, (s_fs+1)//2) 
    
    # list the pixels for which to calculate the error grid
    ijs = []
    for i in np.linspace(roi[0], roi[1]-1, grid[0]):
        for j in np.linspace(roi[2], roi[3]-1, grid[1]):
            ijs.append([round(i), round(j)])
               
    ijs = np.array(ijs).astype(np.int32)
    
    for i in tqdm.trange(1, desc='calculating pixel map shift errors'):
        make_error_map_subpixel(queue, (1, ijs.shape[0]), (1, 1), 
              cl.SVM(Win), 
              cl.SVM(data), 
              localmem, 
              cl.SVM(err_map), 
              cl.SVM(Oin), 
              cl.SVM(pixel_mapout), 
              cl.SVM(dij_nin), 
              cl.SVM(maskin),
              cl.SVM(ijs),
              n0, m0, ijs.shape[0],
              data.shape[0], data.shape[1], data.shape[2], 
              O.shape[0], O.shape[1], ss_min, ss_max, fs_min, fs_max)
         
        queue.finish()
    t1 = time.time()
    t = t1-t0
    
    res = make_pixel_map_err_report(ijs, err_map, mask, search_window, roi, t)
    
    return ijs, err_map, res

def make_pixel_map_err_report(ijs, err_map, mask, search_window, roi, t):
    """
    Should inform the user what the next call to update_pixel_map should be.
    
    We need: 
        suggested grid, 
        suggested window size (I guess this is really a measure of uncertainty)
    """
    # remove masked pixels
    m = mask[ijs[:, 0], ijs[:, 1]]
    
    # report on error land scape:
    # find the average distance between starting point and global min
    ijs_min    = np.argmin(err_map, axis=1)
    i0, j0     = search_window//2, search_window//2
    isol, jsol = np.unravel_index(ijs_min, (search_window, search_window))
    dist       = np.sqrt((i0-isol)**2 + (j0-jsol)**2)
    errs       = np.array([err_map[i, ijs_min[i]] for i in range(err_map.shape[0])])
    med_errs   = np.median(err_map, axis=1)
    with np.errstate(invalid='ignore'):
        rat        = errs / med_errs
    rat_mask = ~np.isnan(rat)
    
    # distance shift / distance pixel
    con = []
    for i in range(ijs[m].shape[0]):
        for j in range(i):
            shift_diff = np.sqrt( (isol[m][i] - isol[m][j])**2 + (jsol[m][i]- jsol[m][j])**2)
            pixel_dist = np.sqrt( (ijs[m][i][0] - ijs[m][j][0])**2 + (ijs[m][i][1] - ijs[m][j][1])**2)
            con.append( shift_diff / pixel_dist)
    con = np.array(con)
    
    print('-----------------------------------------------------------------')
    print('Report on pixel map error landscape: median +- standard deviation')
    print('-----------------------------------------------------------------')
    print('pixel map shift amount                :',np.median(dist[m]), '+-', np.std(dist[m]))
    print('global minimum error                  :',np.median(errs[m]), '+-', np.std(errs[m]))
    print('median error level                    :',np.median(med_errs[m]), '+-',  np.std(med_errs[m]))
    print('minimum error / median err            :',np.median(rat[rat_mask]), '+-',  np.std(rat[rat_mask]))
    print('pixel map shift / physical pixel dist :',np.median(con), '+-',  np.std(con))
    # 
    # remove outliers
    err_thresh = np.median(errs[m])       + 2*np.std(errs[m])
    rat_thresh = np.median(rat[rat_mask]) + 2*np.std(rat[rat_mask])
    rat[~rat_mask] = 2*rat_thresh
    m = mask[ijs[:, 0], ijs[:, 1]] * (errs < err_thresh) * (rat < rat_thresh)
    print('suggested error threshold             :',err_thresh)
    print('suggested ratio threshold             :',rat_thresh)
    
    # recalculate
    rat = errs[m] / med_errs[m]
    con = []
    con_ss = []
    con_fs = []
    for i in range(ijs[m].shape[0]):
        for j in range(i):
            shift_diff_ss = (isol[m][i] - isol[m][j])**2 
            shift_diff_fs = (jsol[m][i] - jsol[m][j])**2 
            pixel_dist = (ijs[m][i][0] - ijs[m][j][0])**2 + (ijs[m][i][1] - ijs[m][j][1])**2 
            con.append( np.sqrt( (shift_diff_ss + shift_diff_fs) / (pixel_dist + pixel_dist) ))
            con_ss.append( np.sqrt( shift_diff_ss / pixel_dist))
            con_fs.append( np.sqrt( shift_diff_fs / pixel_dist))
    con = np.array(con)
    con_ss = np.array(con_ss)
    con_fs = np.array(con_fs)
    
    print('-----------------------------------------------------------------')
    print('After outlier removal (error threshold and ratio threshold)')
    print('-----------------------------------------------------------------')
    print('pixel map shift amount                :',np.median(dist[m]), '+-', np.std(dist[m]))
    print('global minimum error                  :',np.median(errs[m]), '+-', np.std(errs[m]))
    print('median error level                    :',np.median(med_errs[m]), '+-',  np.std(med_errs[m]))
    print('minimum error / median err            :',np.median(rat), '+-',  np.std(rat))
    print('pixel map shift / physical pixel dist :',np.median(con), '+-',  np.std(con))
    
    dist_ss = np.sqrt((i0-isol)**2)
    dist_fs = np.sqrt((j0-jsol)**2)
    # choose window size:
    sug_search_window = [int(round(4*(np.median(dist_ss[m]) + 2*np.std(dist_ss[m])))), 
                         int(round(4*(np.median(dist_fs[m]) + 2*np.std(dist_fs[m]))))]
    
    # choose sampling grid:
    # the tolerable number of pixels between samples is 
    # the number of pixels before con * pixel dist > search_window // 2
    # where search_window is the next smaller search window used for interpolation
    sw_next = 4
    a = (np.median(con_ss) + 2*np.std(con_ss))
    b = (np.median(con_fs) + 2*np.std(con_fs))
    grid = [1, 1]
    if a != 0 :
        pixel_dist_ss = (sw_next // 2) / a
        grid[0] = 2*int(round((roi[1]-roi[0])) / pixel_dist_ss)

    if b != 0 :
        pixel_dist_fs = (sw_next // 2) / b
        grid[1] = 2*int(round((roi[3]-roi[2]) / pixel_dist_fs))
    
    # estimated time
    sec_per_elem = t / (search_window**2 * ijs.shape[0])
    
    res = {'search_window': sug_search_window, 'grid': grid, 'error_threshold': err_thresh}
    print('\n')
    print('suggested search window         :', res['search_window'])
    print('suggested sampling grid         :', res['grid'])
    print('estimated calculation time (min):', round(sec_per_elem * sug_search_window[0]*sug_search_window[1] * grid[0] * grid[1] / 60.,1))
    return res

def bilinear_interpolation_array(array, ss, fs, fill = -1, invalid=-1):
    """
    See https://en.wikipedia.org/wiki/Bilinear_interpolation
    """
    out = np.zeros(ss.shape)
    
    s0, s1 = np.floor(ss).astype(np.uint32), np.ceil(ss).astype(np.uint32)
    f0, f1 = np.floor(fs).astype(np.uint32), np.ceil(fs).astype(np.uint32)
    
    # check out of bounds
    m = (ss > 0) * (ss <= (array.shape[0]-1)) * (fs > 0) * (fs <= (array.shape[1]-1))
    
    s0[~m] = 0
    s1[~m] = 0
    f0[~m] = 0
    f1[~m] = 0
    
    # careful with edges
    s1[(s1==s0)*(s0==0)] += 1
    s0[(s1==s0)*(s0!=0)] -= 1
    f1[(f1==f0)*(f0==0)] += 1
    f0[(f1==f0)*(f0!=0)] -= 1
    
    # make the weighting function
    w00 = (s1-ss)*(f1-fs)
    w01 = (s1-ss)*(fs-f0)
    w10 = (ss-s0)*(f1-fs)
    w11 = (ss-s0)*(fs-f0)
    
    # renormalise for invalid pixels
    w00[array[s0,f0]==invalid] = 0.
    w01[array[s0,f1]==invalid] = 0.
    w10[array[s1,f0]==invalid] = 0.
    w11[array[s1,f1]==invalid] = 0.
    
    # if all pixels are invalid then return fill
    s = w00+w10+w01+w11
    m = (s!=0)*m
    
    out[m] = w00[m] * array[s0[m],f0[m]] \
           + w10[m] * array[s1[m],f0[m]] \
           + w01[m] * array[s0[m],f1[m]] \
           + w11[m] * array[s1[m],f1[m]]
    
    out[m] /= s[m]
    out[~m] = fill
    return out  

def bilinear_interpolation(array, ss, fs, fill = -1, invalid=-1):
    """
    See https://en.wikipedia.org/wiki/Bilinear_interpolation
    """
    import math
    # check out of bounds
    if (ss < 0) or (ss> (array.shape[0]-1)) or (fs < 0) or (fs > (array.shape[1]-1)):
        return fill
    
    s0, s1 = math.floor(ss), math.ceil(ss)
    f0, f1 = math.floor(fs), math.ceil(fs)
    
    # careful with edges
    if s1==s0 :
        if s0 == 0 :
            s1 += 1
        else :
            s0 -= 1
    
    if f1==f0 :
        if f0 == 0 :
            f1 += 1
        else :
            f0 -= 1

    # make the weighting function
    w = np.zeros((2,2), dtype=float)
    a = np.array([[array[s0, f0], array[s0, f1]],
                   array[s1, f0], array[s1, f1]])
    w[0, 0] = (s1-ss)*(f1-fs)
    w[0, 1] = (s1-ss)*(fs-f0)
    w[1, 0] = (ss-s0)*(f1-fs)
    w[1, 1] = (ss-s0)*(fs-f0)
    
    # renormalise for invalid pixels
    w[a==invalid] = 0.
    s = np.sum(w)

    # if all pixels are invalid then return fill
    if s == 0 :
        return fill
    
    w = w/np.sum(w)
    return np.sum( w * a )

def flux_correction(data, W, O, n0, m0, u1, dij_n, mask):
    cs    = [] 
    for n in tqdm.trange(data.shape[0], desc='calculating errors'):
        # define the coordinate mapping and round to int
        ss = u1[0] - dij_n[n, 0] + n0
        fs = u1[1] - dij_n[n, 1] + m0
        #
        I0 = W * bilinear_interpolation_array(O, ss, fs, fill=-1, invalid=-1)
        #
        m = I0>0
        
        c = np.sum(mask * m * I0 * data[n]) / np.sum(mask * m * data[n]**2)
        cs.append(c)
    return (data.T * np.array(cs)).T.astype(np.float32)
