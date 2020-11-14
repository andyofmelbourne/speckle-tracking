import os
import numpy as np
import pyopencl as cl

# do not setup the opencl stuff if we are just building documentation

if os.environ.get('READTHEDOCS') != 'True':
    ##################################################################
    # OpenCL crap
    ##################################################################
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


    # bilinear_interpolation_array
    bilinear_interpolation_array_cl = program.bilinear_interpolation_array

    bilinear_interpolation_array_cl.set_scalar_arg_dtypes(4*[None] + 4 * [np.int32])
        
    # Get the max work group size for the kernel test on our device
    max_comp = device.max_compute_units
    max_size = bilinear_interpolation_array_cl.get_work_group_info(
                       cl.kernel_work_group_info.WORK_GROUP_SIZE, device)

    def bilinear_interpolation_array(array, ss, fs):
        # inputs:
        arrayin     = array.astype(np.float32)
        ssin        = ss.astype(np.float32)
        fsin        = fs.astype(np.float32)

        # outputs:
        out         = np.zeros(ss.shape, dtype=np.float32)
        
        bilinear_interpolation_array_cl(queue, ss.shape, (1, 1), 
                            cl.SVM(arrayin), 
                            cl.SVM(out), 
                            cl.SVM(ssin), cl.SVM(fsin),
                            ss.shape[0], ss.shape[1],
                            arrayin.shape[0], arrayin.shape[1])
        queue.finish()
        
        return out.astype(array.dtype)


    # bilinear_interpolation_inverse_array
    bilinear_interpolation_inverse_array_cl = program.bilinear_interpolation_inverse_array

    bilinear_interpolation_inverse_array_cl.set_scalar_arg_dtypes(4*[None] + 4 * [np.int32])

def bilinear_interpolation_inverse_array(O, I, ss, fs, invalid=-1):
    # output dtype
    d = O.dtype
    
    # inputs:
    Iin         = I.astype(np.float32)
    ssin        = ss.astype(np.float32)
    fsin        = fs.astype(np.float32)
    
    if type(invalid) is np.ndarray :
        Iin[~invalid] = -1
    else :
        Iin[Iin == invalid] = 0
    
    # outputs:
    Oin         = O.astype(np.float32)
    
    #bilinear_interpolation_inverse_array_cl(queue, ss.shape, (1, 1), 
    bilinear_interpolation_inverse_array_cl(queue, (1, 1), (1, 1), 
                        cl.SVM(Iin), 
                        cl.SVM(Oin), 
                        cl.SVM(ssin), cl.SVM(fsin),
                        ss.shape[0], ss.shape[1],
                        O.shape[0], O.shape[1])
    queue.finish()
    
    return Oin.astype(d)


