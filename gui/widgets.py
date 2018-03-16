# set root
import sys, os
directory = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.join(directory, 'widgets'))

from view_h5_data_widget import View_h5_data_widget
