# make a main window with the h5 viewer

try :
    from PyQt5.QtWidgets import *
except :
    from PyQt4.QtGui import *

import signal



# set root
import sys, os
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]

sys.path.insert(0, os.path.join(root, 'utils'))
import cmdline_parser

from view_h5_data_widget import View_h5_data_widget

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
        view_h5_data_widget = View_h5_data_widget(self.fnam)
        self.addTab(view_h5_data_widget, 'view_h5_data_widget')
        
    def closeTab(self, tab):
        self.removeTab(tab)

        

class Speckle_gui(QMainWindow):

    def __init__(self, fnam):
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
        load_view_h5_data_widget.triggered.connect( lambda x : \
                                                    tabs_widget.addTab(View_h5_data_widget(fnam), \
                                                    'view_h5_data_widget') )
        dis_menu.addAction(load_view_h5_data_widget)

        pro_menu = menu.addMenu('Process')

        # auto populate the process menu
        

        self.show()
        

if __name__ == '__main__':
    args, params = cmdline_parser.parse_cmdline_args('speckle-gui', 'speckle-tracking main gui')
    
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    app = QApplication([])
    
    gui = Speckle_gui(args.filename)

    app.exec_()
