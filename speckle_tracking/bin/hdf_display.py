#!/usr/bin/env python
try :
    from PyQt5.QtWidgets import *
except :
    from PyQt4.QtGui import *

from speckle_gui.widgets import show_nd_data_widget


import h5py
import pyqtgraph as pg
import numpy as np
import signal
import argparse

class Hdf_display(QMainWindow):
    def __init__(self, fnam, dataset):
        super(Hdf_display, self).__init__()
        self.setWindowTitle(fnam)
        menu = self.menuBar()
         
        # add the tab widget
        ####################
        widget = show_nd_data_widget.Show_nd_data_widget()
        self.setCentralWidget(widget)
        
        widget.show(fnam, dataset)
        
        self.show()


if __name__ == '__main__':
    description = """
    Display a dataset from a hdf5 file.

    Example:
    hdf_display.py diatom.cxi/speckle_tracking/good_frames 
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('dataset', type=str, \
                        help="file name of the *.cxi followed by the dataset location e.g.: foo.cxi/bar/data")
    
    args = parser.parse_args()

    # now split the dataset name into filename and dataset location
    # assume the last '.' is before the filename extension
    a = args.dataset.split('.')
    fnam    = '.'.join(a[:-1]) + '.' + a[-1].split('/')[0]
    dataset = args.dataset.split(fnam)[-1]

    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    app = QApplication([])
    
    gui = Hdf_display(fnam, dataset)
    
    app.exec_()
