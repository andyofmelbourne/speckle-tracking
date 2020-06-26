#!/usr/bin/env python
import os
import numpy as np
import speckle_tracking as st
from speckle_tracking import cmdline_config_cxi_reader
from speckle_tracking import cmdline_parser 

def str_compare(s, val):
    if type(s) is str and s == val:
        return True
    else :
        return False

if __name__ == '__main__':
    # get command line args and config
    sc  = 'pixel_map_from_data'
    
    # search the current directory for *.ini files if not present in cxi directory
    config_dirs = [os.path.split(os.path.abspath(__file__))[0]]
    
    # extract the first paragraph from the doc string
    des = ''
    
    # get command line args and config params
    args, params = cmdline_config_cxi_reader.get_all(sc, des, config_dirs=config_dirs, roi=False)
    params = params['pixel_map_from_data']
    
    # Initialisation
    #---------------
    
    # make mask
    if str_compare(params['mask'], 'auto'):
        params['mask'] = st.make_mask(params['data'])
        
        # write to file
        out = {'mask': params['mask'], 'h5_group': params['h5_group']}
        cmdline_config_cxi_reader.write_all(out, args.filename, out, apply_roi=False)
        
        # output display for gui
        with open('.log', 'w') as f:
            print('display: '+params['h5_group']+'/mask', file=f)
    
    # make whitefield
    if str_compare(params['whitefield'], 'auto'):
        params['whitefield'] = st.make_whitefield(params['data'], params['mask'])
        
        # write to file
        out = {'whitefield': params['whitefield'], 'h5_group': params['h5_group']}
        cmdline_config_cxi_reader.write_all(out, args.filename, out, apply_roi=False)
        
        # output display for gui
        with open('.log', 'w') as f:
            print('display: '+params['h5_group']+'/whitefield', file=f)
    
    # make ROI
    if str_compare(params['roi'], 'auto'):
        params['roi'] = st.guess_roi(params['whitefield'])
        
        # write to file
        out = {'roi': params['roi'], 'h5_group': params['h5_group']}
        cmdline_config_cxi_reader.write_all(out, args.filename, out, apply_roi=False)
        
        # output display for gui
        with open('.log', 'w') as f:
            print('display: '+params['h5_group']+'/roi', file=f)

    # apply ROI
    shape = params['data'].shape
    roi   = params['roi']
    params['data']        = np.ascontiguousarray(params['data'][:, roi[0]:roi[1], roi[2]:roi[3]])
    params['whitefield']  = params['whitefield'][roi[0]:roi[1], roi[2]:roi[3]]
    params['mask']        = params['mask'][roi[0]:roi[1], roi[2]:roi[3]]
    
    # defocus
    if str_compare(params['defocus'], 'auto'):
        params['defocus'], res= st.fit_thon_rings(
                params['data'],
                params['x_pixel_size'],
                params['y_pixel_size'],
                params['distance'],
                params['wavelength'],
                params['mask'],
                params['whitefield'],
                params['roi'],
                )
        
        # write to file
        res['defocus']  = params['defocus']
        res['h5_group'] = params['h5_group']
        cmdline_config_cxi_reader.write_all(res, args.filename, out, apply_roi=False)
        
        # output display for gui
        with open('.log', 'w') as f:
            print('display: '+params['h5_group']+'/thon_display', file=f)
    
    # pixel map
    if str_compare(params['pixel_map'], 'auto'):
        params['pixel_map'], params['pixel_translations'], res = \
            st.generate_pixel_map(
                params['mask'].shape, 
                params['translations'], 
                params['basis'], 
                params['x_pixel_size'], 
                params['y_pixel_size'], 
                params['z'], 
                params['defocus'], 
                verbose=True)

        # make the 'residual' pixel map for display
        u0 = np.array(np.indices(params['mask'].shape))
        du = params['pixel_map'] - u0
            
        out = {'pixel_map': params['pixel_map'], 
               'pixel_map_residual': du, 
               'pixel_translations': params['pixel_translations'], 
               'roi': params['roi'],
               'h5_group': params['h5_group']}
        out.update(res)
        cmdline_config_cxi_reader.write_all(out, args.filename, out, apply_roi=True)
        
        # output display for gui
        with open('.log', 'w') as f:
            print('display: '+params['h5_group']+'/pixel_map_residual', file=f)


    # make reference
    params['reference_image'], n0, m0 = st.make_reference(
                           params['data'], 
                           params['mask'], 
                           params['whitefield'], 
                           params['pixel_translations'], 
                           params['pixel_map'], 
                           subpixel=True)
    
    out = {'reference_image': params['reference_image'], 
            'n0': n0, 'm0': m0,
            'roi': params['roi'],
            'h5_group': params['h5_group']}
    cmdline_config_cxi_reader.write_all(params, args.filename, out, apply_roi=True)

    # output display for gui
    with open('.log', 'w') as f:
        print('display: '+params['h5_group']+'/reference_image', file=f)
    """
    # make mask
    if params['mask'] == 'auto':
        import make_mask
        make_mask.main(params)

    # make whitefield
    if params['whitefield'] == 'auto':
        import make_whitefield
        make_whitefield.main(params)
                       
    # make ROI
    if params['roi'] == 'auto':
        import guess_roi
        guess_roi.main(params)
    
    # defocus
    if params['defocus'] == 'auto':
        import fit_thon_rings
        fit_thon_rings.main(params)
    
    # pixel map
    if params['pixel_map'] == 'auto':
        import generate_pixel_map
        generate_pixel_map.main(params)
     
    # Main Loop
    #----------
    params0 = copy.deepcopy(params)
    params0['subpixel'] = False
    
    params1 = copy.deepcopy(params)
    params1['subpixel'] = True
    params1['search_window'] = [3,3]
    params1['filter'] = 1.
    
    a = [params0, params1]
    
    for j in range(2):
        params = a[j]
        
        for i in range(3):
            # make reference
            import make_reference
            make_reference.main(params)
            
            # update pixel map
            import update_pixel_map
            update_pixel_map.main(params)
            
            # update translations
            import update_translations
            update_translations.main(params)
            
            # calculate error
            import calc_error
            calc_error.main(params)
    
    # Adiational analysis
    #--------------------
    # calculate sample thickness
    import calculate_sample_thickness
    calculate_sample_thickness.main()
    
    # calculate phase
    import calculate_phase
    calculate_sample_phase.main()

    # calculate focus profile
    import focus_profile
    focus_profile.main()
    """

