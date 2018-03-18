# make a main window with the h5 viewer

try :
    from PyQt5.QtWidgets import *
except :
    from PyQt4.QtGui import *

import signal
import glob

# set the root dir
import sys, os
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]

sys.path.insert(0, os.path.join(root, 'utils'))
import cmdline_parser

import widgets 

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
        tabs_widget = Tabs_widget(fnam)
        self.setCentralWidget(tabs_widget)
        
        # add the display widgets
        dis_menu = menu.addMenu('Display')
        
        # view h5 widget
        load_view_h5_data_widget = QAction("view h5 data widget", self)
        load_view_h5_data_action = lambda x : tabs_widget.addTab(widgets.View_h5_data_widget(fnam), 'view_h5_data_widget')
        load_view_h5_data_widget.triggered.connect( load_view_h5_data_action ) 
        dis_menu.addAction(load_view_h5_data_widget)
        
        pro_menu = menu.addMenu('Process')

        script_names = []
        load_pro_widgets = []
        load_pro_actions = []
            
        # mask maker widget
        script_names.append('mask maker')
        load_pro_widgets.append(QAction(script_names[-1], self))
        load_pro_actions.append(lambda x, s = script_names[-1], 
                                f = fnam : tabs_widget.addTab( \
                                widgets.Mask_maker_widget(f, params['data_path'], params['mask_paths']), s))
        load_pro_widgets[-1].triggered.connect( load_pro_actions[-1] )
        pro_menu.addAction(load_pro_widgets[-1])
        
        # manual tracking widget
        script_names.append('manual_tracking')
        load_pro_widgets.append(QAction(script_names[-1], self))
        load_pro_actions.append(lambda x, s = script_names[-1], f = fnam : tabs_widget.addTab(widgets.Manual_tracking_widget(f), s))
        load_pro_widgets[-1].triggered.connect( load_pro_actions[-1] )
        pro_menu.addAction(load_pro_widgets[-1])
        
        # auto populate the process menu
        ################################
        pro_fnams = glob.glob(root+'/process/*.py')
        for pfnam in pro_fnams:
            script_name = os.path.split(pfnam)[1][:-3]
            
            if script_name in script_names :
                continue

            load_pro_widgets.append(QAction(script_name, self))
            load_pro_actions.append(lambda x, s = script_name, f = fnam : tabs_widget.addTab(widgets.Auto_build_widget(s, f), s))
            load_pro_widgets[-1].triggered.connect( load_pro_actions[-1] )
            pro_menu.addAction(load_pro_widgets[-1])

        self.show()
        
    def autoTab(self, script_name, fnam):
        print(script_name, fnam)

if __name__ == '__main__':
    args, params = cmdline_parser.parse_cmdline_args('speckle-gui', 'speckle-tracking main gui')
    
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    app = QApplication([])
    
    gui = Speckle_gui(args.filename, params)

    app.exec_()
