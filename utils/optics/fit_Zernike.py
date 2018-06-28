from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#from utils.utils import *
import numpy as np

Zernike_index_names = {
        (0, 0)  : "Piston",
        (1, -1) : "tilt y",
        (1, 1)  : "tilt x",
        (2, -2) : "Astigmatism x",
        (2, 0)  : "Defocus",
        (2, 2)  : "Astigmatism y",
        (3, -3) : "Trefoil y",
        (3, -1) : "Primary y coma",
        (3,  1) : "Primary x coma",
        (3,  3) : "Trefoil x",
        (4, -4) : "Tetrafoil y",
        (4, -2) : "Secondary astigmatism y",
        (4,  0) : "Primary spherical",
        (4,  2) : "Secondary astigmatism x",
        (4,  4) : "Tetrafoil x",
        (5,  -5) : "Pentafoil y",
        (5,  -3) : "Secondary tetrafoil y",
        (5,  -1) : "Secondary coma y",
        (5,   1) : "Secondary coma x",
        (5,   3) : "Secondary tetrafoil x",
        (5,   5) : "Pentafoil x",
        (6,   -6) : "",
        (6,   -4) : "Secondary tetrafoil y",
        (6,   -2) : "Tertiary Astigmatism y",
        (6,    0) : "Secondary spherical",
        (6,    2) : "Tertiary Astigmatism y",
        (6,    4) : "Secondary tetrafoil x",
        (6,    6) : "",
        (7, -7) : "",
        (7, -5) : "",
        (7, -3) : "Tertiary trefoil y",
        (7, -1) : "Tertiary coma y",
        (7,  1) : "Tertiary coma x",
        (7,  3) : "Tertiary trefoil x",
        (7,  5) : "",
        (7,  7) : "",
        (8,  0) : "Tertiary spherical",
        }

def make_Zernike_polynomial_cartesian(n, m, order = None):
    """
    Given the Zernike indices n and m return the Zerike polynomial coefficients
    in a cartesian basis.

    The coefficients are stored in a yx matrix of the following form:
    
         1       x        x**2     x**3
    1    yx[0,0] yx[0, 1] yx[0, 2] yx[0, 3]
    y    yx[1,0] yx[1, 1] yx[1, 2] yx[1, 3]
    y**2 yx[2,0] yx[2, 1] yx[2, 2] yx[2, 3] ...
    ...
    
    such that Z^m_n = \sum_i \sum_j yx[i, j] y**i * x**j
    
    yx[i, j] is given by:

    Z^{m}_n  = R^m_n(r) cos(m \theta) 
    Z^{-m}_n = R^m_n(r) sin(m \theta) 
    
    Z^{m}_n  =  \sum_{k=0}^{(n-|m|)/2} (-1)^k (n - k)! / (k! ((n+|m|)/2 -k)! ((n-|m|)/2 -k)!) 
                \sum_{k'=0}^{|m|} binomial(|m|, k') * sin|cos((|m|-k') \pi/2) 
                \sum_{i=0}^{(n-|m|)/2 - k} binomial((n-|m|)/2 - k, i) 
                x^{2i + k'} y^{n - 2k - 2i - k'}
    
    where sin|cos = cos for m >= 0  
    and   sin|cos = sin for m < 0  
    
    Parameters
    ----------
    n, m : int, int
        Zernike indices
    
    order : int
        zero pads yx so that yx.shape = (order, order). If order is less than
        the maximum order of the polynomials then an error will be raised.
    
    Returns 
    -------
    yx : ndarray, int
    
    A : float
        A^m_n, the normalisation factor, e.g. if n = 4 and m = 2 then
        A = \sqrt{10 / \pi}
        
    Reference 
    -------
    For a slightly misguided approach see:
    Efficient Cartesian representation of Zernike polynomials in computer memory
    Hedser van Brug
    SPIE Vol. 3190 0277-786X/97
    """
    from math import factorial as fac
    import math
    
    if (n-m) % 2 == 1 or abs(m) > n or n < 0 :
        return np.array([0]), 0
    
    if m < 0 :
        t0 = math.pi / 2.
    else :
        t0 = 0
    
    m   = abs(m)
    
    if order is None :
        order = n + 1
    
    mat = np.zeros((order, order), dtype=np.int)
    
    for k in range((n-m)//2 + 1):
        a = (-1)**k * fac(n-k) / (fac(k) * fac((n+m)/2 - k) * fac( (n-m)/2 - k))
        for kk in range(m+1):
            b = int(round(math.cos((m-kk)*math.pi/2. - t0)))
            if b is 0 :
                continue
            b *= binomial(m, kk)
            ab = a*b
            l  = (n-m)//2 - k
            for i in range(l + 1):
                c    = binomial(l, i)
                abc  = ab*c
                powx = 2*i + kk
                powy = n - 2*k - 2*i - kk
                mat[powy, powx] += abc

    # compute the normalisation index
    if m is 0 :
        A = math.sqrt( float(n+1) / float(math.pi) )
    else :
        A = math.sqrt( float(2*n+2) / float(math.pi) )
    
    return mat, A

def Gram_Schmit_orthonormalisation(vects, product):
    """
    The following algorithm implements the stabilized Gram-Schmidt orthonormalization.
    
    The vectors 'vects' are replaced by orthonormal vectors which span the same subspace:
        vects = [v0, v1, ..., vN]
        
        u0 = v1
        ...
        uk = vk - \sum_j=0^k-1 proj(uj, vk)

        The basis vectors are then:
        ek = uk / norm(uk)

        where proj(uj, uk) = product(uk, uj) / product(uj, uj) * uj
        and   norm(uk)     = product(uj, uj)

    For the modified algorithm :
        uk = vk - \sum_j=0^k-1 proj(uj, vk)
        
        is replaced by:
                  uk       = uk_{k-2} - proj(uk-1, uk_{k-2})
            where uk_{0}   = vk       - proj(u0, vk)
            and   uk_{k+1} = uk_{k}   - proj(uk, uk_{k})
    
    Parameters
    ----------
    vects : sequence of objects
        The objects in the sequence 'O' must be acceptable to the function 'product'
        and they must have add/subtract/scalar multiplication and scalar division 
        methods.

    product : function of two arguments
        Must take two 'vectors' of type vn and uk and calculate the vector product.
        E.g. product([1,2,3], [4,5,6]) = 1*4 + 2*5 + 3*6. 
    
    Returns
    -------
    basis : sequence of objects
        The orthonormal basis vectors that span the subspace given by 'vects'.
    """
    import math
    import copy
    basis = [vects[0] / math.sqrt(product(vects[0], vects[0]))]
    for k in range(1, len(vects)):
        u = vects[k]
        for j in range(k):
            u = u - basis[j] * product(basis[j], u) 
           
        basis.append(u / math.sqrt(product(u, u)))
    
    return basis

def binomial(N, n):
    """ 
    Calculate binomial coefficient NCn = N! / (n! (N-n)!)

    Reference
    ---------
    PM 2Ring : http://stackoverflow.com/questions/26560726/python-binomial-coefficient
    """
    from math import factorial as fac
    try :
        binom = fac(N) // fac(n) // fac(N - n)
    except ValueError:
        binom = 0
    return binom

def pascal(m):
    """
    Print Pascal's triangle to test binomial()
    """
    for x in range(m + 1):
        print([binomial(x, y) for y in range(x + 1)])

def make_Zernike_polynomial(n, m):
    """
    Given the Zernike indices n and m return the Zerike polynomial coefficients
    for the radial and azimuthal components.

    Z^m_n(r, \theta) = A^n_m cos(m \theta) R^m_n , for m >= 0
    
    Z^m_n(r, \theta) = A^n_m sin(m \theta) R^m_n , for m < 0
    
    R^m_n(r) = \sum_k=0^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n-m)/2 - k)! ((n-m)/2 - k))!} 
               r^{n-2k}

    A^n_m = \sqrt{(2n + 2)/(e_m \pi)}, e_m = 2 if m = 0, e_m = 1 if m != 0

    \iint Z^m_n(r, \theta) Z^m'_n'(r, \theta) r dr d\theta = \delta_{n-n'}\delta_{m-m'}

    Retruns :
    ---------
    p : list of integers
        The polynomial coefficients of R^n_m from largest to 0, e.g. if n = 4 and m = 2 then
        p_r = [4, 0, -3, 0, 0] representing the polynomial 4 r^4 - 3 r^3.

    A : float
        A^m_n, the normalisation factor, e.g. if n = 4 and m = 2 then
        A = \sqrt{10 / \pi}
    """
    if (n-m) % 2 == 1 or abs(m) > n or n < 0 :
        return [0], 0
    
    import math 
    fact = math.factorial 
    p = [0 for i in range(n+1)]

    for k in range((n-abs(m))//2+1):
        # compute the polynomial coefficient for order n - 2k 
        p[n-2*k] = (-1)**k * fact(n-k) / (fact(k) * fact((n+m)/2 - k) * fact((n-m)/2 - k))
    
    # compute the normalisation index
    if m is 0 :
        A = math.sqrt( float(n+1) / float(math.pi) )
    else :
        A = math.sqrt( float(2*n+2) / float(math.pi) )
    
    return p[::-1], A

def make_Zernike_grads(mask, roi = None, max_order = 100, return_grids = False, return_basis = False, yx_bounds = None, test = False):
    if return_grids :
        basis, basis_grid, y, x = make_Zernike_basis(mask, roi, max_order, return_grids, yx_bounds, test)
    else :
        basis = make_Zernike_basis(mask, roi, max_order, return_grids, yx_bounds, test)

    # calculate the x and y gradients
    from numpy.polynomial import polynomial as P
    
    # just a list of [(grad_ss, grad_fs), ...] where the grads are in a polynomial basis
    grads = [ (P.polyder(b, axis=0), P.polyder(b, axis=1)) for b in basis ]

    if return_grids :
        # just a list of [(grad_ss, grad_fs), ...] where the grads are evaluated on a y, x grid
        grad_grids = [(P.polygrid2d(y, x, g[0]), P.polygrid2d(y, x, g[1])) for g in grads]

        if return_basis :
            return grads, grad_grids, basis, basis_grid
        else :
            return grads, grad_grids
    else :
        if return_basis :
            return grads, basis
        else :
            return grads

def make_Zernike_basis(mask, roi = None, max_order = 100, return_grids = False, yx_bounds = None, test = False):
    """
    Make Zernike basis functions, such that:
        np.sum( Z_i * Z_j * mask) = delta_ij

    Returns
    -------
    basis_poly : list of arrays
        The basis functions in a polynomial basis.

    basis_grid : list of arrays
        The basis functions evaluated on the cartesian grid
    """
    shape = mask.shape
    
    # list the Zernike indices in the Noll indexing order:
    # ----------------------------------------------------
    Noll_indices = make_Noll_index_sequence(max_order)
    
    # set the x-y values and scale to the roi
    # ---------------------------------------
    if roi is None :
        roi = [0, shape[0]-1, 0, shape[1]-1]
    
    sub_mask  = mask[roi[0]:roi[1]+1, roi[2]:roi[3]+1] 
    sub_shape = sub_mask.shape
    
    if yx_bounds is None :
        if (roi[1] - roi[0]) > (roi[3] - roi[2]) :
            m = float(roi[1] - roi[0]) / float(roi[3] - roi[2])
            yx_bounds = [-m, m, -1., 1.]
        else :
            m = float(roi[3] - roi[2]) / float(roi[1] - roi[0])
            yx_bounds = [-1., 1., -m, m]
    
    dom = yx_bounds
    y = ((dom[1]-dom[0])*np.arange(shape[0]) + dom[0]*roi[1]-dom[1]*roi[0])/(roi[1]-roi[0])
    x = ((dom[3]-dom[2])*np.arange(shape[1]) + dom[2]*roi[3]-dom[3]*roi[2])/(roi[3]-roi[2])

    # define the area element
    # -----------------------
    dA = (x[1] - x[0]) * (y[1] - y[0])
    
    # generate the Zernike polynomials in a cartesian basis:
    # ------------------------------------------------------
    Z_polynomials = []
    for j in range(1, max_order+1):
        n, m, name           = Noll_indices[j]
        mat, A               = make_Zernike_polynomial_cartesian(n, m, order = max_order)
        Z_polynomials.append(mat * A)
    
    # define the product method
    # -------------------------
    from numpy.polynomial import polynomial as P
    def product(a, b):
        c = P.polygrid2d(y[roi[0]:roi[1]+1], x[roi[2]:roi[3]+1], a)
        d = P.polygrid2d(y[roi[0]:roi[1]+1], x[roi[2]:roi[3]+1], b)
        v = np.sum(dA * sub_mask * c * d)
        return v
    
    basis = Gram_Schmit_orthonormalisation(Z_polynomials, product)
    
    # test the basis function
    if test :
        print('\n\nbasis_i, basis_j, product(basis_i, basis_j)')
        for i in range(len(basis)) :
            for j in range(len(basis)) :
                print(i, j, product(basis[i], basis[j]))

    if return_grids :
        basis_grid = [P.polygrid2d(y, x, b) for b in basis]
        
        if test :
            print('\n\nbasis_i, basis_j, np.sum(mask * basis_i * basis_j)')
            for i in range(len(basis_grid)) :
                for j in range(len(basis_grid)) :
                    print(i, j, np.sum(mask * basis_grid[i] * basis_grid[j]))
        return basis, basis_grid, y, x
    else :
        return basis

def fit_Zernike_coefficients(phase, mask = 1, roi = None, max_order = 100, yx_bounds=None):
    """
    Find cof such that:
        \sum_n cof_n * Z_n[i, j] = phase[i, j]
    
    The Z_n are formed by orthonormalising the Zernike polynomials on the mask.
    The x, y coordinates are scaled and shifted inside the roi such that the 
    smallest dimension is scaled from -1 to 1 and the other in proportion.
    """
    if roi is None :
        roi = [0, shape[0]-1, 0, shape[1]-1]

    if mask is 1 :
        mask = np.ones_like(phase, dtype=np.bool)
    
    sub_mask = np.zeros_like(mask)
    sub_mask[roi[0]:roi[1]+1, roi[2]:roi[3]+1] = mask[roi[0]:roi[1]+1, roi[2]:roi[3]+1]

    basis, basis_grid, y, x = make_Zernike_basis(mask, roi = roi, \
                                           max_order = max_order, return_grids = True, \
                                           yx_bounds = yx_bounds)
    
    Zernike_coefficients = [np.sum(b * sub_mask * phase) for b in basis_grid]
    
    return Zernike_coefficients

def make_Noll_index_sequence(max_j):
    """
    Return a dictionary of tupples where each value is the 
    tupple (n, m), where (n, m) are the Zernike indices, and
    each key is the Noll index.
    
    The natural arrangement of the indices n (radial index) 
    and m (azimuthal index) of the Zernike polynomial Z(n,m) 
    is a triangle with row index n, in each row m ranging from 
    -n to n in steps of 2:
    (0,0)
    (1,-1) (1,1)
    (2,-2) (2,0) (2,2)
    (3,-3) (3,-1) (3,1) (3,3)
    (4,-4) (4,-2) (4,0) (4,2) (4,4)
    (5,-5) (5,-3) (5,-1) (5,1) (5,3) (5,5)
    (6,-6) (6,-4) (6,-2) (6,0) (6,2) (6,4) (6,6)
    (7,-7) (7,-5) (7,-3) (7,-1) (7,1) (7,3) (7,5) (7,7)
    
    For uses in linear algebra related to beam optics, a standard 
    scheme of assigning a single index j>=1 to each double-index 
    (n,m) has become a de-facto standard, proposed by Noll. The 
    triangle of the j at the equivalent positions reads
    1,
    3,2,
    5,4,6,
    9,7,8,10,
    15,13,11,12,14,
    21,19,17,16,18,20,
    27,25,23,22,24,26,28,
    35,33,31,29,30,32,34,36,
    which defines the OEIS entries. The rule of translation is that 
    odd j are assigned to m<0, even j to m>=0, and smaller j to smaller |m|.

    .. math:: Z^m_n(\rho, \theta) = R^m_n(\rho) e^{i m \theta}

    Parameters
    ----------
    max_j : int
        Maximum Noll index for the sequence.
    
    Returns
    -------
    Zernike_indices, dict
        A dictionary pair of {Noll_index : (n, m, name), ...} of length max_j, 
        where Noll_index is an int, and name is a string.

    Refernce
    --------
    https://oeis.org/A176988
    """
    Zernike_indices = {}
    n = 0
    j = 0
    js = []
    nms = []
    while j < max_j :
        # generate the sequence of ms for this row
        ms  = range(-n, n+1, 2)
        
        # append the list (n,m) tupples 
        nms += [(n, m) for m in ms]
        
        # generate the sequence of j's for this row
        jms = range(j+1, j+len(ms)+1)

        # remember the largest value
        j += len(ms)
        
        # assign js largest odd j --> smallest odd j
        js += [j for j in jms[::-1] if j % 2 == 1]
        
        # assign js smallest even j --> largest even j
        js += [j for j in jms if j % 2 == 0]
        
        # increment the row index
        n += 1
    
    # generate the dictionary 
    Zernike_indices = {}
    for j, nm in zip(js, nms):
        if nm in Zernike_index_names.keys() :
            Zernike_indices[j] = nm + (Zernike_index_names[nm],)
        else :
            Zernike_indices[j] = nm + ("",)
    
    return Zernike_indices

if __name__ == '__main__':
    # ----------------------------------------------------------
    # fit Zernike coefficients to a phase profile for arbitrary 
    # aperture dimensions and with masked pixels
    # ----------------------------------------------------------
    print('fiting Zernike coefficients to a phase profile for arbitrary')
    print('aperture dimensions and with masked pixels...')
    shape = (256, 256)
    #roi   = [64, 192, 0, 256]
    roi   = [0, 255, 0, 255]
    
    # stretched domain
    dom_st   = [-1., 1., -1., 1.]
    
    # circle in rectangle domain
    dom_sm   = [-1., 1., -2., 2.]
    
    # rectangle in circle domain
    rat = float(roi[1]-roi[0])/float(roi[3]-roi[2])
    x   = np.sqrt(1. / (1. + rat**2))
    y   = rat * x
    dom_la = [-y, y, -x, x]

    dom = dom_la
    
    #mask  = np.ones(shape, dtype=np.bool)
    mask  = np.random.random( shape ) > 0.2

    # make the phase with the same basis functions as those that are
    # fit, in order to compare coefficients
    Zernike_coefficients = np.random.random((36,))
    basis, basis_grid, y, x = make_Zernike_basis(mask, roi = roi, \
                                           max_order = len(Zernike_coefficients), return_grids = True, \
                                           yx_bounds = dom)
    
    phase = np.sum([Z * b for Z, b in zip(Zernike_coefficients, basis_grid) ], axis=0)
    phase *= mask
    
    fit_coef  = fit_Zernike_coefficients(phase, mask = mask, max_order = 40, roi=roi, yx_bounds=dom)
    
    phase_ret = np.sum([Z * b for Z, b in zip(fit_coef, basis_grid) ], axis=0)
    
    print('Success?')
    print('coefficients == fit coefficients?', np.allclose(Zernike_coefficients, fit_coef[:len(Zernike_coefficients)]))
    print('phase        == fit phase       ?', np.allclose(phase, mask*phase_ret))

    # ----------------------------------------------------------
    # fit Zernike coefficients to phase gradient profiles for 
    # arbitrary aperture dimensions and with masked pixels
    # ----------------------------------------------------------
    #Bdy, Bdx, Bdy_grid, Bdx_grid = make_Zernike_grad(basis, y, x
    grads, grad_grids = make_Zernike_grads(mask, roi = roi, max_order = 36, return_grids = True, yx_bounds = dom, test = False)

