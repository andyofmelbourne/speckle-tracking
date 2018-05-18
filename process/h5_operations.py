import sys, os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root, 'utils'))

import config_reader
import cmdline_parser
import numpy as np
import h5py

if __name__ == '__main__':
    args, params = cmdline_parser.parse_cmdline_args('h5_operations', 'do stuff to h5 files')
    params = params['h5_operations']

    f = h5py.File(args.filename)
    if params['operation'] == 'cp' :
        f.copy(params['from'], params['to'])
        print('cp', params['from'], '-->', params['to']) ; sys.stdout.flush()
    
    elif params['operation'] == 'mv' :
        f.move(params['from'], params['to'])
        print('mv', params['from'], '-->', params['to']) ; sys.stdout.flush()
    
    elif params['operation'] == 'rm' :
        del f[params['from']]
        print('rm:', params['from']) ; sys.stdout.flush()

    f.close()

