__kernel void update_pixel_map(
    __global float  *W,
    __global float  *data,
    __local  float  *data2,
    __global float  *out,
    __global float  *O,
    __global float  *pixel_map,
    __global float  *dij_n,
    __global int    *mask,
    const    float    n0,
    const    float    m0,
    const    int    N,
    const    int    I,
    const    int    J,
    const    int    U,
    const    int    V,
    const    int    di,
    const    int    dj
    )
{                                                       
    int i = get_group_id(0);
    int j = get_group_id(1);
    
    // printf("%i %i %i %i\n", ss_min, ss_max, fs_min, fs_max);
    
    // loop 
    int n, ss, fs;
    float err  = 0.;
    float norm = 0.;
    float t;
    
    for (n = 0; n < N; n++){ 
        data2[n] = data[I*J*n + i*J + j];
    }
    
    if(mask[i*J + j]==1){
            err  = 0.;
            norm = 0.;
            for (n = 0; n < N; n++)
            { 
                ss = rint( pixel_map[0   + i*J + j] + di - dij_n[n*2 + 0] + n0 );
                fs = rint( pixel_map[I*J + i*J + j] + dj - dij_n[n*2 + 1] + m0 );
                
                if((ss >=0) && (ss < U) && (fs >= 0) && (fs < V) && (O[ss*V + fs]>0))
                {
                    t     = data2[n] - W[i*J + j] * O[ss*V + fs];
                    err  += t*t;
                    //
                    t     = data2[n] - W[i*J + j];
                    norm += t*t;
                }
            }
            
            if(norm > 0)
            {
                out[i*J + j] = err / norm;
            }
    }
}

__kernel void update_pixel_map_old(
    __global float  *W,
    __global float  *data,
    __local  float  *data2,
    __global float  *out,
    __global float  *O,
    __global float  *pixel_map,
    __global float  *dij_n,
    __global int    *mask,
    const    float    n0,
    const    float    m0,
    const    int    N,
    const    int    I,
    const    int    J,
    const    int    U,
    const    int    V,
    const    int    ss_min,
    const    int    ss_max,
    const    int    fs_min,
    const    int    fs_max
    )
{                                                       
    int i = get_group_id(0);
    int j = get_group_id(1);

    // printf("%i %i %i %i\n", ss_min, ss_max, fs_min, fs_max);
    
    // loop 
    int n, ss, fs;
    float err  = 0.;
    float norm = 0.;
    float t, err_min = FLT_MAX;
    int i_min, j_min, di, dj;
    
    for (n = 0; n < N; n++){ 
        data2[n] = data[I*J*n + i*J + j];
    }
    
    if(mask[i*J + j]==1){
        for (di = ss_min; di <= ss_max; di++){ 
        for (dj = fs_min; dj <= fs_max; dj++){ 
            err  = 0.;
            norm = 0.;
            for (n = 0; n < N; n++)
            { 
                ss = rint( pixel_map[0   + i*J + j] + di - dij_n[n*2 + 0] + n0 );
                fs = rint( pixel_map[I*J + i*J + j] + dj - dij_n[n*2 + 1] + m0 );
                
                if((ss >=0) && (ss < U) && (fs >= 0) && (fs < V) && (O[ss*V + fs]>0))
                {
                    t     = data2[n] - W[i*J + j] * O[ss*V + fs];
                    err  += t*t;
                    //
                    t     = data2[n] - W[i*J + j];
                    norm += t*t;
                }
            }
            
            if(norm > 0)
            {
                err /= norm;
                
                if(err < err_min)
                {
                    err_min = err;
                    i_min   = di;
                    j_min   = dj;
                }
            }
            
        }}
        
        if(err_min < FLT_MAX){
            out[i*J + j] = err_min;
            pixel_map[0   + i*J + j] += i_min;
            pixel_map[I*J + i*J + j] += j_min;
        }
    }
}

float bilinear_interpolation(
    __global float  *O,
    const    float  ss,
    const    float  fs,
    const    int    V,
    const    int    U
    )
{
    // check out of bounds
    if((ss < 0) || (ss > (U-1)) || (fs < 0) || (fs > (V-1)))
    {
        return -1;
    }
    
    int s0, s1, f0, f1;
    s0 = floor(ss);
    s1 = ceil(ss);
    f0 = floor(fs);
    f1 = ceil(fs);
    
    // careful with edges
    if(s1==s0)
    {
        if(s0 == 0)
        {
            s1 += 1 ;
        }
        else 
        {
            s0 -= 1 ;
        }
    }
    if(f1==f0)
    {
        if(f0 == 0)
        {
            f1 += 1 ;
        }
        else 
        {
            f0 -= 1 ;
        }
    }
    
    float n   = 0.;
    float out = 0.;
    if(O[s0*V + f0] >= 0)
    {
        out += (s1-ss)*(f1-fs)*O[s0*V + f0];
        n   += (s1-ss)*(f1-fs);
    }
    if(O[s0*V + f1] >= 0)
    {
        out += (s1-ss)*(fs-f0)*O[s0*V + f1];
        n   += (s1-ss)*(fs-f0);
    }
    if(O[s1*V + f0] >= 0)
    {
        out += (ss-s0)*(f1-fs)*O[s1*V + f0];
        n   += (ss-s0)*(f1-fs);
    }
    if(O[s1*V + f1] >= 0)
    {
        out += (ss-s0)*(fs-f0)*O[s1*V + f1];
        n   += (ss-s0)*(fs-f0);
    }
    
    // if all pixels are invalid then return fill
    if(n == 0)
    {
        return -1;
    }
    else 
    {
    return out / n;
    }
}

__kernel void update_pixel_map_old_subpixel(
    __global float  *W,
    __global float  *data,
    __local  float  *data2,
    __global float  *out,
    __global float  *O,
    __global float  *pixel_map,
    __global float  *dij_n,
    __global int    *mask,
    const    float    n0,
    const    float    m0,
    const    int    N,
    const    int    I,
    const    int    J,
    const    int    U,
    const    int    V,
    const    int    ss_min,
    const    int    ss_max,
    const    int    fs_min,
    const    int    fs_max,
    const    int    i
    )
{                                                       
    //int i = get_group_id(0);
    int j = get_group_id(1);

    // printf("%i %i %i %i\n", ss_min, ss_max, fs_min, fs_max);
    
    // loop 
    int n; 
    float ss, fs, ot;
    float err  = 0.;
    float norm = 0.;
    float t, err_min = FLT_MAX;
    int i_min, j_min, di, dj;
    
    for (n = 0; n < N; n++){ 
        data2[n] = data[I*J*n + i*J + j];
    }
    
    if(mask[i*J + j]==1){
        for (di = ss_min; di <= ss_max; di++){ 
        for (dj = fs_min; dj <= fs_max; dj++){ 
            err  = 0.;
            norm = 0.;
            for (n = 0; n < N; n++)
            { 
                ss = pixel_map[0   + i*J + j] + di - dij_n[n*2 + 0] + n0;
                fs = pixel_map[I*J + i*J + j] + dj - dij_n[n*2 + 1] + m0;
                
                // evaluate O using bilinear interpolation 
                ot = bilinear_interpolation(O, ss, fs, V, U);
                if(ot >= 0)
                {
                    t     = data2[n] - W[i*J + j] * ot;
                    err  += t*t;
                    //
                    t     = data2[n] - W[i*J + j];
                    norm += t*t;
                }
            }
            
            if(norm > 0)
            {
                err /= norm;
                
                if(err < err_min)
                {
                    err_min = err;
                    i_min   = di;
                    j_min   = dj;
                }
            }
            
        }}
        
        if(err_min < FLT_MAX){
            out[i*J + j] = err_min;
            pixel_map[0   + i*J + j] += i_min;
            pixel_map[I*J + i*J + j] += j_min;
        }
    }
}

