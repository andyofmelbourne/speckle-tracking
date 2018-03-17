import sys, os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root, 'utils'))

import config_reader
import cmdline_parser
import numpy as np

if __name__ == '__main__':
    args, params = cmdline_parser.parse_cmdline_args('test', 'output some crap')
    
    import time
    for i in range(5):
        print(i) ; sys.stdout.flush()
        time.sleep(0.5)
    
    array = np.random.random((256, 256))

    config_reader.write_h5(args.filename, 'test', {'array' : array})
    print('display: test/array') ; sys.stdout.flush()
    
    for i in range(5):
        print(i) ; sys.stdout.flush()
        time.sleep(0.5)
    
    array = np.random.random((256, 256))

    config_reader.write_h5(args.filename, 'test', {'array' : array})
    print('display: test/array') ; sys.stdout.flush()

    for i in range(5):
        print(i) ; sys.stdout.flush()
        time.sleep(0.5)
    
    array = np.random.random((256, 256))

    config_reader.write_h5(args.filename, 'test', {'array' : array})
    print('display: test/array') ; sys.stdout.flush()
