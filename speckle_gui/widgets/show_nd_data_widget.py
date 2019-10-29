try :
    from PyQt5.QtWidgets import *
except :
    from PyQt4.QtGui import *

import h5py
import pyqtgraph as pg
import numpy as np

class Show_nd_data_widget(QWidget):
    def __init__(self):
        super(Show_nd_data_widget, self).__init__()

        self.plotW  = None
        self.plotW2 = None
        self.layout = None
        self.name   = None
        self.initUI()
    
    def initUI(self):
        # set the layout
        self.layout = QVBoxLayout()
        
        # add the layout to the central widget
        self.setLayout(self.layout)
    
    def show(self, filename, name, refresh=False):
        """
        plots:
            (N,)      float, int          --> line plot
            (N, M<4)  float, int          --> line plots
            (N, M>4)  float, complex, int --> 2d image
            (N, M>4)  complex             --> 2d images (abs, angle, real, imag)
            (N, M, L) float, complex, int --> 2d images (real) with slider
        """
        # make plot
        f = h5py.File(filename, 'r')
        shape = f[name].shape

        if self.name == name :
            refresh = True
        
        if not refresh :
            self.close()

        if shape == () :
            if refresh :
                self.plotW.setData(f[name][()])
            else :
                self.plotW = self.text_label = QLabel(self)
                self.plotW.setText('<b>'+name+'</b>: ' + str(f[name][()]))

        elif len(shape) == 1 :
            if refresh :
                self.plotW.setData(f[name][()])
            else :
                self.plotW = pg.PlotWidget(title = name)
                self.plotW.plot(f[name][()], pen=(255, 150, 150))
        
        elif len(shape) == 2 and shape[1] < 4 :
            pens = [(255, 150, 150), (150, 255, 150), (150, 150, 255)]
            if refresh :
                self.plotW.clear()
                for i in range(shape[1]):
                    self.plotW.setData(i, f[name][:, i])
            else :
                self.plotW = pg.PlotWidget(title = name + ' [0, 1, 2] are [r, g, b]')
                for i in range(shape[1]):
                    self.plotW.plot(f[name][:, i], pen=pens[i])

        elif len(shape) == 2 :
            if refresh :
                self.plotW.setImage(f[name][()].astype(np.float).real.T, autoRange = False, autoLevels = False, autoHistogramRange = False)
            else :
                if 'complex' in f[name].dtype.name :
                    title = name + ' (abs, angle, real, imag)'
                else :
                    title = name
                
                frame_plt = pg.PlotItem(title = title)
                self.plotW = pg.ImageView(view = frame_plt)
                self.plotW.ui.menuBtn.hide()
                self.plotW.ui.roiBtn.hide()
                if 'complex' in f[name].dtype.name :
                    im = f[name][()].T.astype(np.float)
                    self.plotW.setImage(np.array([np.abs(im), np.angle(im), im.real, im.imag]))
                else :
                    self.plotW.setImage(f[name][()].astype(np.float).T)

        elif len(shape) == 3 :
            if refresh :
                self.replot_frame()
            else :
                # show the first frame
                frame_plt = pg.PlotItem(title = name)
                self.plotW = pg.ImageView(view = frame_plt)
                self.plotW.ui.menuBtn.hide()
                self.plotW.ui.roiBtn.hide()

                # solve a bug with flat images in pyqtgraph
                for i in range(10):
                    im = f[name][i]
                    if im.max() != im.min() :
                        break
                
                self.plotW.setImage(f[name][i].real.T.astype(np.float).real.T)
                
                # add a little 1d plot with a vline
                self.plotW2 = pg.PlotWidget(title = 'index')
                self.plotW2.plot(np.arange(f[name].shape[0]), pen=(255, 150, 150))
                self.vline = self.plotW2.addLine(x = i, movable=True, bounds = [0, f[name].shape[0]-1])
                self.plotW2.setMaximumSize(10000000, 100)
                    
                self.vline.sigPositionChanged.connect(self.replot_frame)
        
        f.close()
         
        # add to layout
        if refresh is False :
            self.layout.addWidget(self.plotW, stretch = 1)
        
        if self.plotW2 is not None :
            self.layout.addWidget(self.plotW2, stretch = 0)
        
        # remember the last file and dataset for updating
        self.name     = name
        self.filename = filename

    def replot_frame(self):
        i = int(self.vline.value())
        with h5py.File(self.filename, 'r') as f:
            self.plotW.setImage( f[self.name][i].astype(np.float).real.T, autoRange = False, autoLevels = False, autoHistogramRange = False)
    
    def close(self):
        # remove from layout
        if self.layout is not None :
            if self.plotW is not None :
                self.layout.removeWidget(self.plotW)
            
            if self.plotW2 is not None :
                self.layout.removeWidget(self.plotW2)
        
        # close plot widget
        if self.plotW is not None :
            self.plotW.close()
            self.plotW = None
        
        if self.plotW2 is not None :
            self.plotW2.close()
            self.plotW2 = None
    
    def update(self):
        # update the current plot
        self.show(self.filename, self.name, True)
