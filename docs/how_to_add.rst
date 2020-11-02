Adding stuff to the software
****************************

Say that you have a brilliant new idea that you would like to implement in this software suit. It's really great. You want to load the raw data, sum it all up, then write it back into the CXI file: it's called :code:`sum_data.py`:: 

    import numpy as np
    
    def sum_data(data):
        """
        Return the sum of all detector images in the scan.
        """
        out = np.sum(data, axis=0)
        return out

Before you do anything to the code, check out a new branch using git at the top level of the repository (:code:`speckle-tracking`)::
    
    git checkout -b sum_data

Now all changes to the code will be associated with a git branch called :code:`sum_data`.

At this point you have three options: first, you could add this routine to the Python module (in which case the user can call it along with the other speckle tracking routines), second, you could add it as a command line utility (in which case the code does not have to be Python), and third, you could add it to the GUI. 

How to add a routine to the Python module
=========================================

1. Move :code:`sum_data.py` to the directory :code:`speckle-tracking/speckle_tracking`.
2. Add the :code:`sum_data` routine to the file :code:`speckle-tracking/speckle_tracking/__init__.py`::
    
    ...
    # load sum_data into the main module
    from .sum_data import sum_data

3. Commit this code change to your branch::
   
    git add __init__.py
    git add sum_data.py
    git commit -m "my brilliant addition"

4. (Optional) Tell me about your change by emailing it to morganaj@unimelb.edu.au, or by sending a "pull request" at https://github.com/andyofmelbourne/speckle-tracking.


Note that if you have installed this software using :code:`pip intstall -e .`, then no further action is required to use your new routine. 

Now you can open a new terminate, load some data and sum it!::

    >ipython
    import speckle_tracking as st
    import h5py

    with h5py.File('some_cxi_file.cxi', 'r+') as f:
        # read the data from the CXI file
        data = f['/entry_1/data_1/data'][()]

        # call your new function
        result = st.sum_data(data)

        # delete this dataset if it already exists
        if 'my_result/result' in f:
            del f['my_result/result']

        # write the result back into the CXI file
        f['my_result/result'] = result



How to add your routine to the command line
===========================================

1. To make the routine :code:`sum_data` accessible as a command line utility, then first add it as a Python routine above. 

Simple usage
------------
For a basic routine that you don't intend to use as a GUI widget.

2. Write a new file, let's call it :code:`speckle-tracking/speckle_tracking/bin/sum_data_cmd.py`::

    #!/usr/bin/env python
    import sys
    import h5py
    import speckle_tracking as st

    if __name__ == '__main__':
        # grab the CXI file name from the first command line argument
        fnam = sys.args[1]

        with h5py.File(fnam, 'r+') as f:
            # read the data from the CXI file
            data = f['/entry_1/data_1/data'][()]

            # call your new function
            result = st.sum_data(data)

            # delete this dataset if it already exists
            if 'my_result/result' in f:
                del f['my_result/result']

            # write the result back into the CXI file
            f['my_result/result'] = result

3. Commit this code change to your branch::

    git add speckle_tracking/bin/sum_data_cmd.py
    git commit -m "my brilliant command line utility"
    
4. Re-install speckle\_tracking to add :code:`sum_data_cmd.py` to your path. Go to :code:`speckle-tracking` and type::

    pip install -e .

Now you can open a new terminal and run the code with::

    sum_data_cmd.py some_cxi_file.cxi

Complex usage
-------------
If you have many arguments to your routine and you would like to turn this into a GUI widget then you will need two files: one ini file that contains all of the input arguments, and one python file, which calls that ini file and runs your routine.

2a. First create the ini file :code:`speckle-tracking/speckle_tracking/bin/sum_data_cmd.ini`::

    [sum_data_cmd]
    # anything after the ; is a comment
    data  = /entry_1/data_1/data   ;str, location of diffraction data
    
    [sum_data_cmd-advanced]
    h5_group = my_result ;str, name of h5 group to write the result to

2b. Now modify the file :code:`speckle-tracking/speckle_tracking/bin/sum_data_cmd.py`::

    #!/usr/bin/env python
    import sys
    import os
    import h5py
    import speckle_tracking as st

    if __name__ == '__main__':
        # get command line args and config
        sc  = 'sum_data_cmd'
         
        # search the current directory for *.ini files if not present in cxi directory
        config_dirs = [os.path.split(os.path.abspath(__file__))[0]]
        
        # extract the first paragraph from the doc string
        des = st.make_whitefield.__doc__.split('\n\n')[0]
        
        # now load the necessary data
        args, params = st.cmdline_config_cxi_reader.get_all(sc, des, config_dirs=config_dirs)
        
        params = params['sum_data_cmd']
        
        # your data, along with any other options and arguments, 
        # is now in the params dictionary.
        
        # call your new function
        result = st.sum_data(params['data'])
        
        # write the output into CXI file
        out = {'result': result}
        st.cmdline_config_cxi_reader.write_all(params, args.filename, out)

3. Commit this code change to your branch::

    git add speckle_tracking/bin/sum_data_cmd.py
    git add speckle_tracking/bin/sum_data_cmd.ini
    git commit -m "my brilliant command line utility"
    
4. Re-install speckle\_tracking to add :code:`sum_data_cmd.py` to your path. Go to :code:`speckle-tracking` and type::

    pip install -e .

Now you can run the code on the command line with::

    sum_data_cmd.py some_cxi_file.cxi

This will use the default ini file that you have just committed to the code above. After running it, this ini file will be copied into the same directory as the CXI file. If you want to use a different ini file, then use::

    sum_data_cmd.py some_cxi_file.cxi -c some_ini_file.ini

How to add your routine to the GUI
==================================

1. To make the routine :code:`sum_data` accessible in the GUI, first add it as a command line utility above (with the complex syntax).

2. Add the following to :code:`speckle-tracking/speckle_tracking/bin/sum_data_cmd.py`::

    # output display for gui
    with open('.log', 'w') as f:
        print('display: '+params['h5_group']+'/result', file=f)

3. Commit this code change to your branch::

    git add speckle_tracking/bin/sum_data_cmd.py
    git commit -m "my brilliant GUI utility"

That's it! Your routine can now be found in the :code:`Misc` menu of the GUI. It will display the :code:`result` array after the code has finished executing. 

Of course, you can make a custom GUI widget with fancy features, look at the code for the widgets in the :code:`Display` menu to see how this is done. But be warned that it is complicated and tedious, which is why I usually just stick to the auto generated ones. 


