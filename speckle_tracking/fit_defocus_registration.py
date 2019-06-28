import numpy as np
import tqdm

def fit_thon_rings(data, x_pixel_size, y_pixel_size, z, wav, mask, W, roi, centre=None, sig=10, edge_pix=5, window=30, rad_range=None, verbose=True):
def fit_defocus_registration(
        data, x_pixel_size, y_pixel_size, z, 
        wav, mask, W, roi, basis, translations,
        window=30):
    x_pixel_size = np.sqrt(np.sum(basis[0, 0]**2))
    y_pixel_size = np.sqrt(np.sum(basis[0, 1]**2))
    
    # transform the translations onto the ss / fs axes
    # map the translations onto the fs / ss axes
    dx_D = np.array([np.dot(basis[i], translations[i]) for i in range(len(basis))])
    dx_D[:, 0] /= x_pixel_size
    dx_D[:, 1] /= y_pixel_size
    
    # choose the defocus values based on % overlap of adjascent 
    # images 
    # I[n, i] = O[i - r_ss[n] + n0]
    # I[n, i]   = O[i - (x[n]   z / x_pixel_size dzx) + n0]
    # I[n+1, i] = O[i - (x[n+1] z / x_pixel_size dzx) + n0]
    # overlap = (x[n+1] - x[n]) z / (x_pixel_size dzx) - i + (x[n] z / x_pixel_size dzx)
    # left  = i[0]  - (x[n]   z / x_pixel_size dzx) + n0
    # right = i[-1] - (x[n]   z / x_pixel_size dzx) + n0
    # width = i[-1] - i[0]
    # overlap % = |(x[n+1] - x[n]) z / (x_pixel_size dzx) - width| / width
    #           = |(x[n+1] - x[n]) z / (x_pixel_size dzx) - width| / width
    # overlap_min = ((x[n+1] - x[n]) z / (x_pixel_size dzx_max) - width) / width
    #dzs = np.linspace(1e-5, 1000e-6, 1000)
    #overlap = (w - np.abs(dx_D[1, 0] - dx_D[0, 0]) * z / (x_pixel_size * dzs)) / w
    
    # step size (this could break down with some scans...)
    w = roi[1] - roi[0]
    step = np.abs(np.median(np.diff(dx_D[:, 0])))
    overlap_min = 0.5
    overlap_max = 0.98
    dzx_min = step * z / (w * (1 - overlap_min) * x_pixel_size )
    dzx_max = step * z / (w * (1 - overlap_max) * x_pixel_size )
    
    dzs = np.linspace(dzx_min, dzx_max, 100)
    
    # find detector pos corresponding to central region of O
    dx = dzs[-1] * x_pixel_size / z
    dy = dzs[-1] * y_pixel_size / z
    
    m = int(round((data.shape[0]+1)/2.))
    i0 = (roi[1]-roi[0])/2 - np.mean(dx_D[:, 0] - dx_D[m, 0]) / dx
    j0 = (roi[3]-roi[2])/2 - np.mean(dx_D[:, 1] - dx_D[m, 1]) / dy
    
    # sweep of fs (y) defocus
    var = []
    dzx = dzs[-1]
    for k in tqdm.trange(len(dzs), desc='sweeping over fs defocus'):
        dzy = dzs[k]
        O, overlap = make_O_subregion(dzx, dzy, roi, z, x_pixel_size, y_pixel_size, dx_D, mask, data, m, i0, j0, window=100)
        var.append(np.var(O[O>0]) * np.mean(overlap[O>0]))
    
    # quadratic correction # v = a i**2 + b i + c # vmax = v[-2b / a]
    i = np.argmax(var)
    p = np.polyfit(dzs[i-1:i+2], var[i-1:i+2], 2)
    dzy = -p[1]/(2*p[0])

    # sweep of ss (x) defocus
    var = []
    for k in tqdm.trange(len(dzs), desc='sweeping over ss defocus'):
        dzx = dzs[k]
        O, overlap = make_O_subregion(dzx, dzy, roi, z, x_pixel_size, y_pixel_size, dx_D, mask, data, m, i0, j0, window=100)
        var.append(np.var(O[O>0]) * np.mean(overlap[O>0]))
    
    # quadratic correction # v = a i**2 + b i + c # vmax = v[-2b / a]
    i = np.argmax(var)
    p = np.polyfit(dzs[i-1:i+2], var[i-1:i+2], 2)
    dzx = -p[1]/(2*p[0])
    
    O, overlap = make_O_subregion(dzx, dzy, roi, z, x_pixel_size, y_pixel_size, dx_D, mask, data, m, i0, j0, window=100)

    z1 = (dzx + dzy)/2.
    return z1, {'O_subregion': O, 'defocus_fs': dzy, 'defocus_ss': dzx, 'astigmatism': (dzx-dzy)/2.}

def make_O_subregion(dzx, dzy, roi, z, x_pixel_size, y_pixel_size, 
                     dx_D, mask, data, m, i0, j0, window=100):
    dx = dzx * x_pixel_size / z
    dy = dzy * y_pixel_size / z
    
    dij_n = dx_D[:, :2].copy()
    dij_n[:, 0] /= dx
    dij_n[:, 1] /= dy
    
    # choose the offset so that ij - dij_n + n0 > -0.5
    n0, m0 = np.max(dij_n[:, 0])-0.5, np.max(dij_n[:, 1])-0.5
    
    # data[n, i, j] = O[i - dij_n[n, 0] + n0, 
    #                   j - dij_n[n, 1] + m0] 
    # suppose we choose one pixel in the data say: n, i, j = m, i0, j0
    # 
    # data[m, i0, j0] = O[io, jo]
    # io = i0 - dij_n[m, 0] + n0
    # jo = j0 - dij_n[m, 1] + m0
    
    w = window//2
    io = i0 - dij_n[m, 0] + n0
    jo = j0 - dij_n[m, 1] + m0
    
    io_min = int(round(io-w))
    jo_min = int(round(jo-w))
    
    m = np.ones_like(mask)
    ij = np.indices(mask.shape) 
    
    # determine the object-map domain
    shape   = (window+1, window+1)
    I       = np.zeros(shape, dtype=np.float)
    overlap = np.zeros(shape, dtype=np.float)
    WW      = W**2
    minimum_overlap = 2
    
    for n in tqdm.trange(data.shape[0], desc='building object map'):
        i = int(round(io + dij_n[n, 0] - n0))
        j = int(round(jo + dij_n[n, 1] - m0))
        i_min = max(i-w, 0)
        j_min = max(j-w, 0)
        i_max = max(i+w, 0)
        j_max = max(j+w, 0)
        m.fill(False)
        m[i_min:i_max, j_min:j_max] = mask[i_min:i_max, j_min:j_max]
        # define the coordinate mapping and round to int
        ss = np.rint((ij[0][m] - dij_n[n, 0] + n0)).astype(np.int) - io_min
        fs = np.rint((ij[1][m] - dij_n[n, 1] + m0)).astype(np.int) - jo_min
        #
        I[      ss, fs] += (W*data[n])[m]
        overlap[ss, fs] += WW[m]
            
    overlap[overlap<1e-2] = -1
    m = (overlap > 0)
    
    I[m]  = I[m] / overlap[m]
    I[~m] = -1
    
    if minimum_overlap is not None :
        m = overlap < minimum_overlap
        I[m] = -1
    return I[:-1, :-1], overlap[:-1, :-1]
