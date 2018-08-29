# set root
import sys, os
directory = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.join(directory, 'widgets'))

from view_h5_data_widget          import View_h5_data_widget
from auto_build_widget            import Auto_build_widget
from mask_maker_widget            import Mask_maker_widget
from show_frames_selection_widget import Show_frames_selection_widget 
from update_pixel_map_widget      import Update_pixel_map_widget 
from fit_defocus_widget           import Fit_defocus_widget
#from grad_descent_widget          import Grad_descent_widget 
#from manual_tracking_widget       import Manual_tracking_widget
