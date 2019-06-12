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

def integrate(dss, dfs, mask, maxiter=3000):
    def grad_ss(p):
        return p[1:, :] - p[:-1, :]

    def grad_fs(p):
        return p[:, 1:] - p[:, :-1]

    def f(x):
        return np.sum( mask * (grad_ss(x)[:,:-1]-dss)**2 ) + np.sum( mask * (grad_fs(x)[:-1,:]-dfs)**2 )

    def df(x):
        out = np.zeros_like(x)
        #
        # outi,j       = [xi,j       - xi-1,j      - dssi-1,j ] mi-1,j
        out[1:, :-1]  += (x[1:, :-1] - x[:-1, :-1] - dss[:, :])*mask
        #
        # outi,j       = [xi+1,j     - xi,j        - dssi,j   ] mi,j
        out[:-1, :-1] -= (x[1:, :-1] - x[:-1, :-1] - dss[:, :])*mask
        #
        # outi,j       = [xi,j       - xi,j-1      - dssi,j-1 ] mi,j-1
        out[:-1, 1:]  += (x[:-1, 1:] - x[:-1, :-1] - dfs[:, :])*mask
        #
        # outi,j       = [xi,j+1       - xi,j      - dssi,j   ] mi,j
        out[:-1, :-1] -= (x[:-1, 1:] - x[:-1, :-1] - dfs[:, :])*mask
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
                    d[1:,:-1]  * d[1:,:-1]  -  d[1:,:-1]  * d[:-1,:-1] \
                   -d[:-1,:-1] * d[1:,:-1]  +  d[:-1,:-1] * d[:-1,:-1] \
                   +d[:-1,1:]  * d[:-1,1:]  -  d[:-1,1:]  * d[:-1,:-1] \
                   -d[:-1,:-1] * d[:-1,1:]  +  d[:-1,:-1] * d[:-1,:-1]
                    )*mask)
        return out
    
    #assert(np.allclose(df(phi), 0*phi))
    
    fd = lambda x, d : np.sum( d * df(x))
    
    tx = np.zeros((dss.shape[0]+1, dss.shape[1]+1), dtype=float)
    ty = np.zeros((dss.shape[0]+1, dss.shape[1]+1), dtype=float)
    
    tx[1:,:-1]  = np.cumsum(dss, axis=0)
    tx  = tx - tx[tx.shape[0]//2, :]
    ty[1:,:-1]  = np.cumsum(dfs, axis=1)
    ty  = (ty.T - ty[:, ty.shape[1]//2]).T
    
    x0 = (tx + ty) / 2.
    
    cgls = Cgls(x0, f, df, fd, dfd=dfd, imax=maxiter)
    phase = cgls.cgls()

    dss_forw = grad_ss(phase)[:,:-1] 
    dfs_forw = grad_fs(phase)[:-1,:] 
    
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
        import tqdm
        t = tqdm.trange(iterations, desc='cgls err:')
        for i in t:
            #
            # perform a line search of f along d
            self.x, status = self.line_search(self.x, self.d)
            if status is False :
                print('Warning: line search failed!', t1, t2)
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
            if (self.iters % 50 == 0) or (status == False) :
                self.d = self.r
            #
            # calculate the error
            self.errors.append(self.f(self.x))
            self.iters = self.iters + 1
            if self.iters > self.imax or (self.errors[-1] < self.e_tol):
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
