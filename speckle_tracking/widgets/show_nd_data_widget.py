try :
    from PyQt5.QtWidgets import *
except :
    from PyQt4.QtGui import *

import h5py
import pyqtgraph as pg
import numpy as np

# test
import threading

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

        scatter:
            (N, M, 2) float, int          --> N overlayed scatter plots
        """
        if len(name.split(' ')) == 2 :
            name, im_type = name.split(' ')
        else :
            im_type = None
        
        # make plot
        f = h5py.File(filename, 'r')
        shape = f[name].shape
        title = name + ' ' + str(shape) + ' ' + str(f[name].dtype)

        if self.name == name :
            refresh = True
        elif (self.name is not None) and (self.name != name):
            self.close()

        #if not refresh :
        #    self.close()

        if shape == () :
            if refresh :
                self.plotW.setText('<b>'+name+'</b>: ' + str(f[name][()]))
            else :
                self.plotW = self.text_label = QLabel(self)
                self.plotW.setText('<b>'+title+'</b>: ' + str(f[name][()]))

        elif len(shape) == 1 :
            if refresh :
                self.plotW.clear()
            else :
                self.plotW = pg.PlotWidget(title = title)

            self.plotW.plot(f[name][()], pen=(255, 150, 150))
        
        elif (len(shape) == 1) or (len(shape) == 2 and shape[0]<5) : 
            if len(shape) == 1 :
                I = [()]
            else :
                I = range(shape[0])
                np.random.seed(3)
            
            pen = (255, 150, 150)
            
            if refresh :
                self.plotW.clear()
            else :
                self.plotW = pg.PlotWidget(title = title)
            
            for i, ii in enumerate(I):
                if ii>0:
                    pen = tuple(np.random.randint(0, 255, 3))
                
                self.plotW.plot(f[name][i], pen=pen)
                print('pen', pen)

        elif (len(shape) == 2 or len(shape) == 3) and im_type == 'scatter' : 
            if refresh :
                self.plotW.clear()
            else :
                self.plotW = pg.PlotWidget(title = title)
            
            self.ss = []
            # scatter plot
            ##############
            if len(shape) == 2 :
                X = f[name][:, 0]
                Y = f[name][:, 1]
                pen   = pg.mkPen((255, 150, 150))
                brush = pg.mkBrush(255, 255, 255, 120)
                 
                self.s1    = pg.ScatterPlotItem(size=5, pen=pen, brush=brush)
                spots = [{'pos': [X[n], Y[n]], 'data': n} for n in range(len(X))] 
                self.s1.addPoints(spots)
                self.ss.append(self.s1)
            else :
                np.random.seed(3)
                for n in range(f[name].shape[0]):
                    X = f[name][n, :, 0]
                    Y = f[name][n, :, 1]

                    if n == 0 :
                        pen   = pg.mkPen((255, 150, 150))
                        brush = pg.mkBrush(255, 255, 255, 120)
                    else :
                        pen   = pg.mkPen(tuple(np.random.randint(0, 255, 3)))
                        brush = pg.mkBrush(tuple(np.random.randint(0, 255, 4)))
                     
                    self.ss.append(pg.ScatterPlotItem(size=5, pen=pen, brush=brush))
                    spots = [{'pos': [X[n], Y[n]], 'data': n} for n in range(len(X))] 
                    self.ss[-1].addPoints(spots)
            
            for s1 in self.ss:
                self.plotW.addItem(s1)
        
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
                    title = title + ' (abs, angle, real, imag)'
                else :
                    title = title
                
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
                # show the middle frame
                frame_plt = pg.PlotItem(title = title)
                self.plotW = pg.ImageView(view = frame_plt)
                self.plotW.ui.menuBtn.hide()
                self.plotW.ui.roiBtn.hide()
                
                i = f[name].shape[0]//2
                
                # set min / max to the 10 and 90'th percentile
                im = f[name][i]
                minl = np.percentile(im, 10.)
                maxl = np.percentile(im, 90.)

                self.plotW.setImage(im.astype(np.float).real.T, levels=(minl, maxl))

                # set min max of histogram widget to minl and maxl
                hw = self.plotW.getHistogramWidget()
                hw.item.setHistogramRange(minl, maxl)

                
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
