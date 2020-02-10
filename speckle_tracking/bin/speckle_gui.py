#!/usr/bin/env python
# make a main window with the h5 viewer

try :
    from PyQt5.QtWidgets import *
except :
    from PyQt4.QtGui import *

import sys, os
import signal
import glob
import numpy as np


import speckle_tracking as st
from speckle_tracking import cmdline_parser

root = os.path.abspath(os.path.split(st.__file__)[0])

widgets = st._widgets 
#from . import widgets 
print(os)

class Tabs_widget(QTabWidget):
    def __init__(self, fnam):
        super(Tabs_widget, self).__init__()
        self.fnam = fnam
        self.initUI()
    
    def initUI(self):
        self.setMovable(True)
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self.closeTab)

        # by default load the h5-viewer
        view_h5_data_widget = widgets.View_h5_data_widget(self.fnam)
        self.addTab(view_h5_data_widget, 'view_h5_data_widget')
        
    def closeTab(self, tab):
        self.removeTab(tab)


class Speckle_gui(QMainWindow):

    def __init__(self, fnam, params):
        super(Speckle_gui, self).__init__()
        self.setWindowTitle(fnam)
        menu = self.menuBar()
         
        # add the tab widget
        ####################
        tabs_widget = Tabs_widget(fnam)
        self.setCentralWidget(tabs_widget)
        
        # add the display widgets
        dis_menu = menu.addMenu('Display')
        
        # Show frames tab
        #################
        def load_sfs(fnam, R_paths = params['translation_paths'], W_paths=params['whitefield_paths'],
                     data_paths = params['data_paths'], good_frames_paths=params['good_frames_paths']):
            tabs_widget.addTab(widgets.Show_frames_selection_widget(fnam, R_paths, W_paths, data_paths, good_frames_paths), 
                                          'show / select frames')
        
        load_sfs_widget = QAction("show / select frames", self)
        load_sfs_action = lambda x : load_sfs(fnam)
        load_sfs_widget.triggered.connect( load_sfs_action ) 
        dis_menu.addAction(load_sfs_widget)
        load_sfs_action(fnam)
        
        # view h5 widget
        ################
        load_view_h5_data_widget = QAction("view h5 data widget", self)
        load_view_h5_data_action = lambda x : tabs_widget.addTab(widgets.View_h5_data_widget(fnam), 'view_h5_data_widget')
        load_view_h5_data_widget.triggered.connect( load_view_h5_data_action ) 
        dis_menu.addAction(load_view_h5_data_widget)
        
        script_names = []

        # update_pixel_map widget
        #########################
        #script_names.append('update_pixel_map')
        #load_pro_widgets.append(QAction(script_names[-1], self))
        #load_pro_actions.append(lambda x, s = script_names[-1], f = fnam : tabs_widget.addTab(widgets.Update_pixel_map_widget(s, f), s))
        #load_pro_widgets[-1].triggered.connect( load_pro_actions[-1] )
        #pro_menu.addAction(load_pro_widgets[-1])

        # update_pixel_map widget
        #########################
        #script_names.append('pos_refine')
        #load_pro_widgets.append(QAction(script_names[-1], self))
        #load_pro_actions.append(lambda x, s = script_names[-1], f = fnam : tabs_widget.addTab(widgets.Update_pixel_map_widget(s, f), s))
        #load_pro_widgets[-1].triggered.connect( load_pro_actions[-1] )
        #pro_menu.addAction(load_pro_widgets[-1])
        
        # fit_defocus widget
        #########################
        #script_names.append('fit_defocus_thon')
        #load_pro_widgets.append(QAction(script_names[-1], self))
        #load_pro_actions.append(lambda x, s = script_names[-1], 
        #                        f = fnam : tabs_widget.addTab(widgets.Fit_defocus_widget(s, f), s))
        #load_pro_widgets[-1].triggered.connect( load_pro_actions[-1] )
        #pro_menu.addAction(load_pro_widgets[-1])

        # populate the Initialisation menu
        ##################################
        init_menu = menu.addMenu('Initialisation')
        
        load_init_widgets = []
        load_init_actions = []

        # mask maker widget
        ###################
        script_names.append('mask maker')
        load_init_widgets.append(QAction(script_names[-1], self))
        load_init_actions.append(lambda x, s = script_names[-1], 
                                f = fnam : tabs_widget.addTab( \
                                widgets.Mask_maker_widget(f, params['data_paths'], params['mask_paths']), s))
        load_init_widgets[-1].triggered.connect( load_init_actions[-1] )
        init_menu.addAction(load_init_widgets[-1])
        
        # auto populate from list
        init_fnams = "make_mask.py make_whitefield.py guess_roi.py fit_defocus_registration.py fit_thon_rings.py generate_pixel_map.py".split()
        
        for init_fnam in init_fnams :
            print(init_fnam)
            script_name = os.path.split(init_fnam)[1][:-3]
            
            mpi = False 
            
            script_names.append(script_name)
            load_init_widgets.append(QAction(script_name, self))
            load_init_actions.append(lambda x, s = script_name, f = fnam : tabs_widget.addTab(widgets.Auto_build_widget(s, f, mpi=mpi), s))
            load_init_widgets[-1].triggered.connect( load_init_actions[-1] )
            init_menu.addAction(load_init_widgets[-1])
        
        # populate the Main Loop menu
        ##################################
        main_menu = menu.addMenu('Main Loop')
        
        # auto populate from list
        main_fnams = "make_reference.py update_pixel_map.py calc_error.py update_translations.py".split()
        
        load_main_widgets = []
        load_main_actions = []
        for main_fnam in main_fnams :
            print(main_fnam)
            script_name = os.path.split(main_fnam)[1][:-3]
            
            mpi = False 
            
            script_names.append(script_name)
            load_main_widgets.append(QAction(script_name, self))
            load_main_actions.append(lambda x, s = script_name, f = fnam : tabs_widget.addTab(widgets.Auto_build_widget(s, f, mpi=mpi), s))
            load_main_widgets[-1].triggered.connect( load_main_actions[-1] )
            main_menu.addAction(load_main_widgets[-1])


        # populate the Additional Analysis menu
        ##################################
        add_menu = menu.addMenu('Additional Analysis')
        
        # auto populate from list
        add_fnams = "calculate_phase.py calculate_sample_thickness.py focus_profile.py split_half_recon.py zernike.py".split()
        
        load_add_widgets = []
        load_add_actions = []
        for add_fnam in add_fnams :
            print(add_fnam)
            script_name = os.path.split(add_fnam)[1][:-3]
            
            mpi = False 
            
            script_names.append(script_name)
            load_add_widgets.append(QAction(script_name, self))
            load_add_actions.append(lambda x, s = script_name, f = fnam : tabs_widget.addTab(widgets.Auto_build_widget(s, f, mpi=mpi), s))
            load_add_widgets[-1].triggered.connect( load_add_actions[-1] )
            add_menu.addAction(load_add_widgets[-1])
        
        # auto populate the misc menu
        ################################
        misc_menu = menu.addMenu('Misc')
        
        load_misc_widgets = []
        load_misc_actions = []
            
        misc_fnams = glob.glob(root+'/bin/*.py')
        exclude = ['write_h5.py', 'wipe_cxi.py', 'speckle_gui.py', 'hdf_display.py']
        
        for pfnam in misc_fnams :
            print(pfnam)
            if np.any([e in pfnam for e in exclude]) :
                continue 
            
            script_name = os.path.split(pfnam)[1][:-3]
            
            if script_name in script_names :
                continue
            
            mpi = False 
            
            script_names.append(script_name)
            load_misc_widgets.append(QAction(script_name, self))
            load_misc_actions.append(lambda x, s = script_name, f = fnam : tabs_widget.addTab(widgets.Auto_build_widget(s, f, mpi=mpi), s))
            load_misc_widgets[-1].triggered.connect( load_misc_actions[-1] )
            misc_menu.addAction(load_misc_widgets[-1])
        
        self.show()
        
    def autoTab(self, script_name, fnam):
        print(script_name, fnam)


def main():
    args, params = cmdline_parser.parse_cmdline_args('speckle_gui', 
                                                     'speckle-tracking main gui', 
                                                     config_dirs=[os.path.dirname(__file__),])
    
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    app = QApplication([])
    
    gui = Speckle_gui(args.filename, params['speckle-gui'])
    
    app.exec_()
    
if __name__ == '__main__':
    main()
