__kernel void update_pixel_map(
    __global float  *W,
    __global float  *data,
    __local  float  *data2,
    __global float  *out,
    __global float  *O,
    __global float  *pixel_map,
    __global float  *dij_n,
    __global int    *mask,
    __global int    *is,
    __global int    *js,
    const    float    n0,
    const    float    m0,
    const    float  subsample,
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
    int i, j;
    j = get_group_id(1);
    i = is[j];
    j = js[j];
    
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
        for (di = ss_min; di < ss_max; di++){ 
        for (dj = fs_min; dj < fs_max; dj++){ 
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

__kernel void update_pixel_map_split(
    __global float  *W,
    __global float  *data,
    __local  float  *data2,
    __global float  *out,
    __global float  *O,
    __global float  *pixel_map,
    __global float  *dij_n,
    __global int    *mask,
    __local  int    *mask2,
    __global int    *is,
    __global int    *js,
    const    float    n0,
    const    float    m0,
    const    float  subsample,
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
    int i, j;
    j = get_group_id(1);
    i = is[j];
    j = js[j];
    
    // printf("%i %i %i %i\n", ss_min, ss_max, fs_min, fs_max);
    
    // loop 
    int n, ss, fs;
    float err  = 0.;
    float norm = 0.;
    float t, err_min = FLT_MAX;
    int i_min, j_min, di, dj;
    
    for (n = 0; n < N; n++){ 
        data2[n] = data[I*J*n + i*J + j];
        mask2[n] = mask[I*J*n + i*J + j];
    }
    
    for (di = ss_min; di < ss_max; di++){ 
    for (dj = fs_min; dj < fs_max; dj++){ 
        err  = 0.;
        norm = 0.;
        for (n = 0; n < N; n++)
        { 
            if(mask2[n]==1){
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


float bilinear_interpolation(
    __global float  *O,
    const    float  ss,
    const    float  fs,
    const    int    V,
    const    int    U
    )
{
    int s0, s1, f0, f1;
    s0 = floor(ss);
    s1 = ceil(ss);
    f0 = floor(fs);
    f1 = ceil(fs);
    
    // careful with edges
    if(s1==s0)
    {
        s1 += 1 ;
    }
    if(f1==f0)
    {
        f1 += 1 ;
    }
    
    float n   = 0.;
    float out = 0.;
    if((s0>=0) && (s0<U) && (f0>=0) && (f0<V) && (O[s0*V + f0] >= 0))
    {
        out += (s1-ss)*(f1-fs)*O[s0*V + f0];
        n   += (s1-ss)*(f1-fs);
    }
    if((s0>=0) && (s0<U) && (f1>=0) && (f1<V) && (O[s0*V + f1] >= 0))
    {
        out += (s1-ss)*(fs-f0)*O[s0*V + f1];
        n   += (s1-ss)*(fs-f0);
    }
    if((s1>=0) && (s1<U) && (f0>=0) && (f0<V) && (O[s1*V + f0] >= 0))
    {
        out += (ss-s0)*(f1-fs)*O[s1*V + f0];
        n   += (ss-s0)*(f1-fs);
    }
    if((s1>=0) && (s1<U) && (f1>=0) && (f1<V) && (O[s1*V + f1] >= 0))
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

__kernel void bilinear_interpolation_inverse_array(
    __global float  *array,
    __global float  *out,
    __global float  *ss,
    __global float  *fs,
    const    int    I,
    const    int    J,
    const    int    U,
    const    int    V
    )
{
    // out[ss[i, j], fs[i, j]] += array[i, j]
    // shape of ss, fs and array = I, J 
    // shape of out = U, V
    //int i = get_group_id(0);
    //int j = get_group_id(1);
    int i, j;
    for (i = 0; i < I; i++){ 
    for (j = 0; j < J; j++){ 
    
    // check for invalid
    if (array[i*J + j] >= 0)
    {
    
    int s0, s1, f0, f1;
    s0 = floor(ss[i*J + j]);
    s1 = ceil(ss[i*J + j]);
    f0 = floor(fs[i*J + j]);
    f1 = ceil(fs[i*J + j]);
    
    // careful with edges
    if(s1==s0)
    {
        s1 += 1 ;
    }
    if(f1==f0)
    {
        f1 += 1 ;
    }
    
    //printf("%i %i %f %f\n", s0, s1, ss[i*J+j], fs[i*J+j]);
    if((s0>=0) && (s0<U) && (f0>=0) && (f0<V))
    {
        out[s0*V + f0] += (s1-ss[i*J + j])*(f1-fs[i*J + j])*array[i*J + j];
        //printf("%i %i %i %i %f\n", i, j, s0, f0, array[i*J + j]);
    }
    if((s0>=0) && (s0<U) && (f1>=0) && (f1<V))
    {
        out[s0*V + f1] += (s1-ss[i*J + j])*(fs[i*J + j]-f0)*array[i*J + j];
    }
    if((s1>=0) && (s1<U) && (f0>=0) && (f0<V))
    {
        out[s1*V + f0] += (ss[i*J + j]-s0)*(f1-fs[i*J + j])*array[i*J + j];
    }
    if((s1>=0) && (s1<U) && (f1>=0) && (f1<V))
    {
        out[s1*V + f1] += (ss[i*J + j]-s0)*(fs[i*J + j]-f0)*array[i*J + j];
    }

    }
    }}
}


__kernel void bilinear_interpolation_array(
    __global float  *array,
    __global float  *out,
    __global float  *ss,
    __global float  *fs,
    const    int    I,
    const    int    J,
    const    int    U,
    const    int    V
    )
{                                                       
    int i = get_group_id(0);
    int j = get_group_id(1);

    out[i*J + j] = bilinear_interpolation(array, ss[i*J+j], fs[i*J + j], V, U);
}



__kernel void pixel_map_err_split(
    __global float  *W,
    __global float  *data,
    __local  float  *data2,
    __global float  *out,
    __global float  *O,
    __global float  *pixel_map,
    __global float  *dij_n,
    __global int    *mask,
    __local  int    *mask2,
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
    int n;
    float err  = 0.;
    float norm = 0.;
    float t, ot, ss, fs;
    
    for (n = 0; n < N; n++){ 
        data2[n] = data[I*J*n + i*J + j];
        mask2[n] = mask[I*J*n + i*J + j];
    }
    
    err  = 0.;
    norm = 0.;
    for (n = 0; n < N; n++)
    { 
        if(mask2[n]==1){
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
    }
    
    if(norm > 0)
    {
        out[i*J + j] = err / norm;
    }
}

__kernel void update_pixel_map_subpixel(
    __global float  *W,
    __global float  *data,
    __local  float  *data2,
    __global float  *out,
    __global float  *O,
    __global float  *pixel_map,
    __global float  *dij_n,
    __global int    *mask,
    __global int    *is,
    __global int    *js,
    const    float    n0,
    const    float    m0,
    const    float  subsample,
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
    //int i = get_group_id(0);
    int i, j;
    j = get_group_id(1);
    i = is[j];
    j = js[j];

    // printf("%i %i %i %i\n", ss_min, ss_max, fs_min, fs_max);
    
    // loop 
    int n; 
    float ss, fs, ot;
    float err  = 0.;
    float norm = 0.;
    float t, err_min = FLT_MAX;
    float i_min, j_min, di, dj;
    
    for (n = 0; n < N; n++){ 
        data2[n] = data[I*J*n + i*J + j];
    }
    
    if(mask[i*J + j]==1){
        for (di = ss_min; di < ss_max; di+=1./subsample){ 
        for (dj = fs_min; dj < fs_max; dj+=1./subsample){ 
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
                
                // printf("%i %f %f %f %f\n", j, di, dj, err_min, err);
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
            //printf("%i %i %f %f\n", i, j, i_min, j_min);
        }
    }
}

// return the error map for a subset of pixels
// ijs[i, 0] = ss pixel
// ijs[i, 1] = fs pixel
// ijs.shape = (II, 2)
// out[i, ss_shift, fs_shift] = error at pixel ss, fs for ss_shift, fs_shift
// out.shape = (II, ss_max-ss_min+1, fs_max-fs_min+1)
__kernel void make_error_map_subpixel(
    __global float  *W,
    __global float  *data,
    __local  float  *data2,
    __global float  *out,
    __global float  *O,
    __global float  *pixel_map,
    __global float  *dij_n,
    __global int    *mask,
    __global int    *ijs,
    const    float    n0,
    const    float    m0,
    const    int    II,
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
    //int ii = get_group_id(0);
    int jj = get_group_id(1);
    
    int i, j;
    
    // ijs.shape = (II, 2)
    // jj : 0 --> 2*II
    i = ijs[2*jj + 0];
    j = ijs[2*jj + 1];
    
    // printf("%i %i %i %i\n", ss_min, ss_max, fs_min, fs_max);
    
    // loop 
    int n, l, o; 
    float ss, fs, ot;
    float err  = 0.;
    float norm = 0.;
    float t;
    int di, dj;
    
    l = ss_max - ss_min;
    o = fs_max - fs_min;
    
    for (n = 0; n < N; n++){ 
       data2[n] = data[I*J*n + i*J + j];
    }

    if(mask[i*J + j]==1){
       for (di = ss_min; di < ss_max; di++){ 
       for (dj = fs_min; dj < fs_max; dj++){ 
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
               out[jj*l*o + o*(di-ss_min) + dj-fs_min] = err;
           }
           else 
           {
               out[jj*l*o + o*(di-ss_min) + dj-fs_min] = -1.;
           }
           
           
       }}
      
    }
}

__kernel void pixel_map_err(
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
    int n;
    float err  = 0.;
    float norm = 0.;
    float t, ot, ss, fs;
    
    for (n = 0; n < N; n++){ 
        data2[n] = data[I*J*n + i*J + j];
    }
    
    if(mask[i*J + j]==1){
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
                
                //if((ss >=0) && (ss < U) && (fs >= 0) && (fs < V) && (O[ss*V + fs]>0))
                //{
                //    t     = data2[n] - W[i*J + j] * O[ss*V + fs];
                //    err  += t*t;
                    //
                //    t     = data2[n] - W[i*J + j];
                //    norm += t*t;
                //}
            }
            
            if(norm > 0)
            {
                out[i*J + j] = err / norm;
            }
    }
}


__kernel void translations_err(
    __global float  *W,
    __global float  *data,
    __global float  *O,
    __global float  *pixel_map,
    __global float  *dij_n,
    __global int    *mask,
    __global int    *ns,
    __global float  *out,
    __global float  *di,
    __global float  *dj,
    const    float    n0,
    const    float    m0,
    const    int    I,
    const    int    J,
    const    int    U,
    const    int    V
    )
{                                                       
    int n = get_group_id(0);
    n = ns[n];
    
    
    // loop 
    int i, j;
    float err  = 0.;
    float norm  = 0.;
    float t, ot, ss, fs;
    
    for (i = 0; i < I; i++){ 
    for (j = 0; j < J; j++){ 
    if(mask[i*J + j]==1){
        ss = pixel_map[0   + i*J + j] - di[n] - dij_n[n*2 + 0] + n0;
        fs = pixel_map[I*J + i*J + j] - dj[n] - dij_n[n*2 + 1] + m0;
        
        // evaluate O using bilinear interpolation 
        ot = bilinear_interpolation(O, ss, fs, V, U);
        if(ot >= 0)
        {
            t     = data[I*J*n + i*J + j] - W[i*J + j] * ot;
            err  += t*t;
            //
            t     = data[I*J*n + i*J + j] - W[i*J + j];
            norm += t*t;
        }
    }}}
    if(norm>0){
        out[n] = err/norm;
    }
}
