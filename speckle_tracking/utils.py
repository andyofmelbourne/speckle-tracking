import numpy as np
import h5py

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

def integrate(dssin, dfsin, maskin, maxiter=3000, stepin=[1.,1.]):
    # first rescale the problem so that values are ~1
    scale = np.mean(np.array([dssin, dfsin]))
    dss = dssin / scale
    dfs = dfsin / scale
    mask = maskin / np.mean(maskin)
    step = [1, stepin[1]/stepin[0]]
    
    def grad_ss(p):
        return (p[1:, :] - p[:-1, :])/step[0]

    def grad_fs(p):
        return (p[:, 1:] - p[:, :-1])/step[1]

    def f(x):
        return np.sum( mask * (grad_ss(x)[:,:-1]-dss)**2 ) + np.sum( mask * (grad_fs(x)[:-1,:]-dfs)**2 )

    def df(x):
        out = np.zeros_like(x)
        #
        # outi,j       = [xi,j       - xi-1,j      - dssi-1,j ] mi-1,j
        out[1:, :-1]  += ((x[1:, :-1] - x[:-1, :-1])/step[0]**2 - dss[:, :])*mask
        #
        # outi,j       = [xi+1,j     - xi,j        - dssi,j   ] mi,j
        out[:-1, :-1] -= ((x[1:, :-1] - x[:-1, :-1])/step[0]**2 - dss[:, :])*mask
        #
        # outi,j       = [xi,j       - xi,j-1      - dfsi,j-1 ] mi,j-1
        out[:-1, 1:]  += ((x[:-1, 1:] - x[:-1, :-1])/step[1]**2 - dfs[:, :])*mask
        #
        # outi,j       = [xi,j+1       - xi,j      - dfsi,j   ] mi,j
        out[:-1, :-1] -= ((x[:-1, 1:] - x[:-1, :-1])/step[1]**2 - dfs[:, :])*mask
        return 2*out 

    def dfd(x, d):
        out = 0.
        #
        # out += di,j di,j mi-1,j 
        #
        # out -= di,j di-1,j mi-1,j
        #
        # out -= di,j di+1,j mi,j
        #
        # out += di,j di,j mi,j
        #
        # out += di,j di,j mi,j-1
        #
        # out -= di,j di,j-1 mi,j-1
        #
        # out -= di,j di,j+1 mi,j
        #
        # out += di,j di,j mi,j
        out = 2*np.sum((
                    d[1:,:-1]  * d[1:,:-1]/step[0]**2  -  d[1:,:-1]  * d[:-1,:-1]/step[0]**2 \
                   -d[:-1,:-1] * d[1:,:-1]/step[0]**2  +  d[:-1,:-1] * d[:-1,:-1]/step[0]**2 \
                   +d[:-1,1:]  * d[:-1,1:]/step[1]**2  -  d[:-1,1:]  * d[:-1,:-1]/step[1]**2 \
                   -d[:-1,:-1] * d[:-1,1:]/step[1]**2  +  d[:-1,:-1] * d[:-1,:-1]/step[1]**2
                    )*mask)
        return out
    
    #assert(np.allclose(df(phi), 0*phi))
    
    fd = lambda x, d : np.sum( d * df(x))
    
    tx = np.zeros((dss.shape[0]+1, dss.shape[1]+1), dtype=float)
    ty = np.zeros((dss.shape[0]+1, dss.shape[1]+1), dtype=float)
    
    tx[1:,:-1]  = step[0] * np.cumsum(dss, axis=0)
    #tx[:, 0]  = tx[:, 1] 
    #tx[:, -1] = tx[:, -2] 
    tx  = tx - tx[tx.shape[0]//2, :]
    ty[1:,:-1]  = step[1] * np.cumsum(dfs, axis=1)
    #ty[:, 0]  = ty[:, 1] 
    #ty[:, -1] = ty[:, -2] 
    ty  = (ty.T - ty[:, ty.shape[1]//2]).T
    
    x0 = (tx + ty) / 2.
    x0[0, :]  = x0[1, :]
    x0[:, -1] = x0[1, -2]
    
    cgls = Cgls(x0, f, df, fd, dfd=dfd, imax=maxiter, e_tol = 1e-4)

    scale *= stepin[0]
    
    phase = scale * cgls.cgls()

    dss_forw = grad_ss(phase)[:,:-1]/stepin[0] 
    dfs_forw = grad_fs(phase)[:-1,:]/stepin[0]
    
    return phase[:-1, :-1], {'dss_forward': dss_forw,
                             'dfs_forward': dfs_forw,
                             'dss_residual': dss_forw-dss,
                             'dfs_residual': dfs_forw-dfs,
                             'cgls_errors': cgls.errors}

def integrate_grad2(dssin, dfsin, maskin, maxiter=3000, stepin=[1.,1.], sigma=1.):
    scale = np.mean(np.array([dssin, dfsin]))
    dss = dssin / scale
    dfs = dfsin / scale
    mask = maskin / np.mean(maskin)
    step = [1, stepin[1]/stepin[0]]
    
    def grad_ss(p):
        return (p[1:, :] - p[:-1, :])/step[0]
    
    def grad_fs(p):
        return (p[:, 1:] - p[:, :-1])/step[1]
    
    def grad2_ss(p):
        return (p[2:, :] - 2*p[1:-1, :] + p[:-2, :])/step[0]**2
    
    def grad2_fs(p):
        return (p[:, 2:] - 2*p[:, 1:-1] + p[:, :-2])/step[1]**2
    
    def f(x):
        return np.sum( mask * (grad_ss(x)[:,:-1]-dss)**2 ) \
             + np.sum( mask * (grad_fs(x)[:-1,:]-dfs)**2 ) \
             + sigma * np.sum( grad2_ss(x)**2) \
             + sigma * np.sum( grad2_fs(x)**2) 

    def f_grad(x):
        return np.sum( mask * (grad_ss(x)[:,:-1]-dss)**2 ) \
             + np.sum( mask * (grad_fs(x)[:-1,:]-dfs)**2 )

    def f_grad2(x):
        return sigma * np.sum( grad2_ss(x)**2) \
             + sigma * np.sum( grad2_fs(x)**2) 

    def df(x):
        out = np.zeros_like(x)
        t = (x[2:,:] - 2*x[1:-1,:] + x[:-2,:])/step[0]**4
        #
        # out_i,j     +=  sig [xi,j - 2xi-1,j + xi-2,j]
        out[2:,:] += t
        #
        # out_i,j     -= 2sig [xi+1,j - 2xi,j + xi-1,j]
        out[1:-1,:] -= 2*t
        #
        # out_i,j     +=  sig [xi+2,j - 2xi+1,j + xi,j]
        out[:-2,:] += t
        #
        # out_i,j     +=  sig [xi,j - 2xi,j-1 + xi,j-2]
        t = (x[:,2:] - 2*x[:,1:-1] + x[:,:-2])/step[1]**4
        out[:,2:] += t
        #
        # out_i,j     -= 2sig [xi,j+1 - 2xi,j + xi,j-1]
        out[:,1:-1] -= 2* t
        #
        # out_i,j     +=  sig [xi,j+2 - 2xi,j+1 + xi,j]
        out[:,:-2] += t
        #
        out *= sigma
        #
        # outi,j       = [xi,j       - xi-1,j      - dssi-1,j ] mi-1,j
        t = ((x[1:, :-1] - x[:-1, :-1])/step[0] - dss[:, :])*mask/step[0]
        out[1:, :-1]  += t
        #
        # outi,j       = [xi+1,j     - xi,j        - dssi,j   ] mi,j
        out[:-1, :-1] -= t
        #
        # outi,j       = [xi,j       - xi,j-1      - dfsi,j-1 ] mi,j-1
        t = ((x[:-1, 1:] - x[:-1, :-1])/step[1] - dfs[:, :])*mask/step[1]
        out[:-1, 1:]  += t
        #
        # outi,j       = [xi,j+1       - xi,j      - dfsi,j   ] mi,j
        out[:-1, :-1] -= t
        return 2*out 


    def dfd(x, d):
        out = 0.
        #
        # out += di,j di,j mi-1,j 
        #
        # out -= di,j di-1,j mi-1,j
        #
        # out -= di,j di+1,j mi,j
        #
        # out += di,j di,j mi,j
        #
        # out += di,j di,j mi,j-1
        #
        # out -= di,j di,j-1 mi,j-1
        #
        # out -= di,j di,j+1 mi,j
        #
        # out += di,j di,j mi,j
        out = 2*np.sum((
                    d[1:,:-1]  * d[1:,:-1]/step[0]**2  -  d[1:,:-1]  * d[:-1,:-1]/step[0]**2 \
                   -d[:-1,:-1] * d[1:,:-1]/step[0]**2  +  d[:-1,:-1] * d[:-1,:-1]/step[0]**2 \
                   +d[:-1,1:]  * d[:-1,1:]/step[1]**2  -  d[:-1,1:]  * d[:-1,:-1]/step[1]**2 \
                   -d[:-1,:-1] * d[:-1,1:]/step[1]**2  +  d[:-1,:-1] * d[:-1,:-1]/step[1]**2
                    )*mask)
        #
        t = d[2:,:] - 2*d[1:-1,:] + d[:-2,:]
        out += 2*sigma*np.sum(
                 d[2:,:]   * t \
              -2*d[1:-1,:] * t \
                +d[:-2,:]  * t \
               ) / step[0]**4
        #
        t = d[:,2:] - 2*d[:, 1:-1] + d[:, :-2]
        out += 2*sigma*np.sum(
                 d[:,2:]   * t \
              -2*d[:,1:-1] * t \
                +d[:,:-2]  * t \
               ) / step[1]**4
        return out

    fd = lambda x, d : np.sum( d * df(x))

    cgls = Cgls(np.zeros((dss.shape[0]+1, dss.shape[1]+1)), 
                f, df, fd, dfd=dfd, imax=maxiter, e_tol = 1e-10)
    
    scale *= stepin[0]
    
    phase = scale * cgls.cgls()

    dss_forw = grad_ss(phase)[:,:-1]/stepin[0] 
    dfs_forw = grad_fs(phase)[:-1,:]/stepin[0]
    
    return phase[:-1, :-1], {'dss_forward': dss_forw,
                             'dfs_forward': dfs_forw,
                             'dss_residual': dss_forw-dss,
                             'dfs_residual': dfs_forw-dfs,
                             'cgls_errors': cgls.errors}

def line_search_newton_raphson(x, d, fd, dfd, iters = 1, tol=1.0e-10):
    """Finds the minimum of the the function f along the direction of d by using a second order Taylor series expansion of f.

    f(x + alpha * d) ~ f(x) + alpha * f'(x) . d + alpha^2 / 2 * dT . f''(x) . d
    therefore alpha = - fd / dfd 
    #
    fd  is a function that evaluates f'(x) . d
    dfd is a function that evaluates dT . f''(x) . d
    #
    returns x2, status
    #
    status is True if dfd is > tol and False otherwise.
    if status is False then the local curvature is negative and the 
    minimum along the line is infinitely far away.
    Algorithm from Eq. (57) of painless-conjugate-gradient.pdf
    """
    for i in range(iters):
        fd_i  = fd(x, d)
        dfd_i = dfd(x, d)
        if dfd_i < tol :
            return x, False
        #
        alpha = - fd_i / dfd_i 
        #
        x = x + alpha * d
    return x, True

class Cgls(object):
    """Minimise the function f using the nonlinear cgls algorithm.
    
    """

    def __init__(self, x0, f, df, fd, dfd = None, imax = 10**5, e_tol = 1.0e-10):
        self.f     = f
        self.df    = df
        self.fd    = fd
        self.iters = 0
        self.imax  = imax
        self.e_tol = e_tol
        self.errors = []
        self.x     = x0
        if dfd is not None :
            self.dfd         = dfd
            self.line_search = lambda x, d: line_search_newton_raphson(x, d, self.fd, self.dfd, iters = 1, tol=1.0e-10)
        else :
            self.dfd         = None
            self.line_search = lambda x, d: line_search_secant(x, d, self.fd, iters = 20, sigma = 1.0, tol=1.0e-10)
        #
        #self.cgls = self.cgls_Ploak_Ribiere
        self.cgls = self.cgls_Flecher_Reeves

    def cgls_Flecher_Reeves(self, iterations = None):
        """
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iters == 0 :
            self.r         = - self.df(self.x)
            self.d         = self.r.copy()
            self.delta_new = np.sum(self.r**2)
            self.delta_0   = self.delta_new.copy()
        # 
        if iterations == None :
            iterations = self.imax
        #
        derr = 1
        import tqdm
        t = tqdm.trange(iterations, desc='cgls err:')
        for i in t:
            #
            # perform a line search of f along d
            self.x, status = self.line_search(self.x, self.d)
            if status is False :
                print('Warning: line search failed!')
            # 
            self.r         = - self.df(self.x)
            delta_old      = self.delta_new
            self.delta_new = np.sum(self.r**2)
            #
            # Fletcher-Reeves formula
            beta           = self.delta_new / delta_old
            #
            self.d         = self.r + beta * self.d
            #
            # reset the algorithm 
            if (self.iters % 10000 == 0) or (status == False) :
                self.d = self.r
            #
            # calculate the error
            self.errors.append(self.f(self.x))
            self.iters = self.iters + 1
            if len(self.errors) > 1 :
                derr = (self.errors[-2] - self.errors[-1]) / self.errors[-1]
            
            if self.iters > self.imax or (derr < self.e_tol):
                t.close()
                break
            t.set_description("cgls err: {:.2e}".format(self.errors[-1]))
        #
        #
        return self.x

    def cgls_Ploak_Ribiere(self, iterations = None):
        """
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iters == 0 :
            self.r         = - self.df(self.x)
            self.r_old     = self.r.copy()
            self.d         = self.r.copy()
            self.delta_new = np.sum(self.r**2)
            self.delta_0   = self.delta_new.copy()
        # 
        if iterations == None :
            iterations = self.imax
        #
        for i in range(iterations):
            #
            # perform a line search of f along d
            self.x, status = self.line_search(self.x, self.d)
            # 
            self.r         = - self.df(self.x)
            delta_old      = self.delta_new
            delta_mid      = np.sum(self.r * self.r_old)
            self.r_old     = self.r.copy()
            self.delta_new = np.sum(self.r**2)
            #
            # Polak-Ribiere formula
            beta           = (self.delta_new - delta_mid)/ delta_old
            #
            self.d         = self.r + beta * self.d
            #
            # reset the algorithm 
            if (self.iters % 50 == 0) or (status == False) or beta <= 0.0 :
                self.d = self.r
            else :
                self.d = self.r + beta * self.d
            #
            # calculate the error
            self.errors.append(self.f(self.x))
            self.iters = self.iters + 1
            if self.iters > self.imax or (self.errors[-1] < self.e_tol):
                break
        #
        #
        return self.x

def bilinear_interpolation_array(array, ss, fs, fill = -1, invalid=-1):
    """
    ss, fs = slow and fast scan coordinates in pixel units
    
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

def write_h5(write, fnam = 'siemens_star.cxi', og = 'speckle_tracking/'):
    with h5py.File(fnam) as f:
        for k in write.keys():
            if (og+k) in f :
                del f[og+k]
            f[og+k] = write[k]

def read_h5(read, fnam = 'siemens_star.cxi', og = 'speckle_tracking/'):
    out = {}
    with h5py.File(fnam) as f:
        for k in read:
            out[k] = f[og+k][()]
    return out
