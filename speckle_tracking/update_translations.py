"""
__kernel void update_translations(
    __global float  *W,
    __global float  *data,
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
                    norm ++;
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
"""
