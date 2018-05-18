try :
    from PyQt5.QtWidgets import *
except :
    from PyQt4.QtGui import *

import pyqtgraph as pg
import h5py
import numpy as np
import itertools

from select_frames_widget import Select_frames_widget

def white(f, path, i):
    Ws = f[path]
    if len(Ws.shape) == 2 :
        W = Ws[()].astype(np.float)
    elif len(Ws.shape) == 3 :
        W = Ws[i].astype(np.float)
    W[W<=0] = 1
    return W

class Show_frames_selection_widget(QWidget):
    def __init__(self, filename, R_paths=None, W_paths=None, data_paths=None, good_frames_paths=None):
        super(Show_frames_selection_widget, self).__init__()
        
        self.filename = filename
        f = h5py.File(self.filename, 'r')
        
        # load the data shape
        for data_path in data_paths :
            if data_path in f :
                self.shape = f[data_path].shape
                self.D_path = data_path
                break
        
        # load the good frames
        self.good_frames = np.arange(self.shape[0])
        for g_path in good_frames_paths :
            if g_path in f :
                self.good_frames = f[g_path][()]
                break

        # load the translations
        for R_path in R_paths :
            if R_path in f :
                self.R = f[R_path][()]
                break
        
        # load the whitefield
        W = 1
        if W_paths is not None :
            for W_path in W_paths :
                if W_path in f :
                    self.W_path = W_path
                    break
        
        f.close()
        
        self.filename = filename
        self.initUI()

    def initUI(self):
        # Make a grid layout
        layout = QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # X and Y plot 
        ##############
        R = self.R
        X = R[:, 0]
        Y = R[:, 1]
        title = 'realspace x (red) and y (green) sample positions in pixel units'
        position_plotsW = pg.PlotWidget(bottom='frame number', left='position', title = title)
        position_plotsW.plot(X, pen=(255, 150, 150))
        position_plotsW.plot(Y, pen=(150, 255, 150))
        
        # vline
        vline = position_plotsW.addLine(x = 0, movable=True, bounds = [0, len(X)-1])

        # frame plot
        ############
        f = h5py.File(self.filename, 'r')
        self.mkframe = lambda i, f : f[self.D_path][i] / white(f, self.W_path, i)
        
        frame_plt = pg.PlotItem(title = 'Frame View')
        imageView = pg.ImageView(view = frame_plt)
        imageView.ui.menuBtn.hide()
        imageView.ui.roiBtn.hide()
        imageView.setImage(self.mkframe(0, f).T.astype(np.float))
        #imageView.show()

        # scatter plot
        ##############
        ## 1: X/YPZT
        self.scatter_plot = Select_frames_widget(self.filename, self.good_frames, self.R)
        
        layout.addWidget(imageView      , 0, 0, 1, 1)
        layout.addWidget(self.scatter_plot   , 0, 1, 1, 1)
        layout.addWidget(position_plotsW, 1, 0, 1, 2)
        layout.setColumnMinimumWidth(0, 800)
        layout.setRowMinimumHeight(0, 500)

        j = 0
        def replot_frame():
            i = int(vline.value())
            self.scatter_plot.replot(i)

            f = h5py.File(self.filename, 'r')
            imageView.setImage( self.mkframe(i, f).T.astype(np.float), autoRange = False, autoLevels = False, autoHistogramRange = False)
            f.close()
            
        vline.sigPositionChanged.connect(replot_frame)
        f.close()
