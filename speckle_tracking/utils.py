import numpy as np

def mk_2dgaus(shape, sig, centre = None):
    if centre is None :
        centre = [shape[0]//2, shape[1]//2]
    if sig is not None : 
        x = np.arange(shape[0]) - centre[0]
        x = np.exp( -x**2 / (2. * sig**2))
        y = np.arange(shape[1]) - centre[1]
        y = np.exp( -y**2 / (2. * sig**2))
        reg = np.outer(x, y)
    else :
        reg = 1
    return reg

def get_r_theta(shape, is_fft_shifted = True):
    i = np.fft.fftfreq(shape[0]) * shape[0]
    j = np.fft.fftfreq(shape[1]) * shape[1]
    i, j = np.meshgrid(i, j, indexing='ij')
    rs   = np.sqrt(i**2 + j**2)
    
    ts = np.arctan2(i, j)
    if is_fft_shifted is False :
        rs = np.fft.fftshift(rs)
        ts = np.fft.fftshift(ts)
    return rs, ts

def radial_symetry(background, rs, mask=None):
    ########### Find the radial average
    # mask zeros from average
    if mask is None : 
        mask = np.ones(background.shape, dtype=np.bool)

    rs = rs.astype(np.int16).ravel()
    
    # get the r histogram
    r_hist = np.bincount(rs, mask.ravel().astype(np.int16))
    # get the radial total 
    r_av = np.bincount(rs, background.ravel() * mask.ravel())
    # prevent divide by zero
    nonzero = np.where(r_hist != 0)
    zero    = np.where(r_hist == 0)
    # get the average
    r_av[nonzero] = r_av[nonzero] / r_hist[nonzero].astype(r_av.dtype)
    r_av[zero]    = 0
    
    ########### Make a large background filled with the radial average
    #background = r_av[rs].reshape(background.shape)
    return r_av

def diff_inv_2d(da, a0=None, axis=0):
    shape        = list(da.shape)
    shape[axis] += 1
    out          = np.zeros(tuple(shape), dtype=float)
    if a0 is None :
        if axis == 0 :
            a0 = np.zeros((shape[1],))
        else :
            a0 = np.zeros((shape[0],))
        
    if axis == 0 :
        out[0,  :] = a0
        out[1:, :] = da
    elif axis == 1 :
        out[:,  0] = a0
        out[:, 1:] = da
    return np.cumsum(out, axis=axis)

def P(g, df, mask, axis):
    """
    h(x, y) = cumsum(f) = f(x, y) + c(y)
    er = sum [g-h]^2
       = sum_ij [g_ij - f_ij - c_j]^2
    er_gk = sum_i -2 [g_ik - f_ik - c_k] = 0
    therefore c_j = sum_i [g_ik - f_ik] / I
    """
    df2 = df.copy()
    #df2[~mask] = np.diff(g, axis=axis)[~mask]
    df2[~mask] = 0.
    if axis == 0 :
        h  = diff_inv_2d(df2, a0 = g[:,0], axis=0)
        #h += np.mean(g-h, axis=0)
        # the masked region of g should remain untouched
        h[1:, :][~mask] = g[1:, :][~mask]
        # now we have to adjust the mean level of each 
        # segment (between masked pixels) to that of g
        # this could take a while... should be opencl
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):


                    
    elif axis == 1 :
        h = diff_inv_2d(df2, a0 = None, axis=1)
        h = (h.T + np.mean(g-h, axis=1)).T
    return h

def integrate(dss, dfs, mask, tol=1e-10, maxiter=1000):
    s = dss.shape
    dfx = np.zeros((s[0], s[1]+1), dtype=float)
    dfy = np.zeros((s[0]+1, s[1]), dtype=float)
    mx  = np.zeros(dfx.shape, dtype=np.bool)
    my  = np.zeros(dfy.shape, dtype=np.bool)
    mx[:, :-1] = mask
    my[:-1, :] = mask
    dfx[:, :-1] = dss
    dfy[:-1, :] = dfs
    norm = np.sum(dss**2 + dfs**2)
    
    f = np.zeros((s[0]+1, s[1]+1), dtype=float)
    er  = np.sqrt((np.sum((mask*(dss-np.diff(f, axis=0)[:,:-1]))[:-1,:]**2) + np.sum((mask*(dfs-np.diff(f, axis=1)[:-1,:] )**2))[:,:-1])/norm)
    print(-1, er)
    for i in range(maxiter):
        fx = P(f, dfx, mx, 0)
        fy = P(f, dfy, my, 1)
        #fy = fx.copy()
        f = (fx + fy)/2.
        #fx = P(f, dfx, mx, 0)
        #fy = P(fx, dfy, my, 1)
        #f = fy.copy()
        er  = np.sqrt((np.sum((mask*(dss-np.diff(f, axis=0)[:,:-1]))[:-1,:]**2) + np.sum((mask*(dfs-np.diff(f, axis=1)[:-1,:] )**2))[:,:-1])/norm)
        er2 = np.sqrt(np.sum(mask*(fx-fy)[:-1,:-1]**2)/norm)
        er3 = (np.median(mx*(np.diff(fx, axis=0)-dfx)**2), np.median(my*(np.diff(fy, axis=1)-dfy)**2))
        er4 = (np.sum(mask*(f-fx)[:-1,:-1]**2), np.sum(mask*(f-fy)[:-1,:-1]**2))
        #er2 = np.sqrt(np.sum(mask*(fx-fy)[:-1,:-1]**2)/norm)
        #er3 = (np.sqrt(np.sum(mask*(dss-np.diff(f, axis=0)[:,:-1])**2)/norm),  np.sum(np.sqrt(mask*(dfs-np.diff(f, axis=1)[:-1,:] )**2)/norm))
        print(i, er2, er3, er4, er)
        if er < tol :
            break
    return f
