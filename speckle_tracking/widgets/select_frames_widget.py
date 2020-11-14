try :
    import PyQt5.QtWidgets as pyqt
except :
    import PyQt4.QtGui as pyqt

import h5py
import pyqtgraph as pg
import numpy as np

class Select_frames_widget(pyqt.QWidget):
    """
    Draw a scatter plot of the X-Y coordinates in f[R]
    """
        
    def __init__(self, filename, good_frames, R, frame=0):
        super(Select_frames_widget, self).__init__()
        self.R = R
        self.good_frames = good_frames
        
        self.filename = filename
        self.frames = []
        self.initUI(frame)
    
    def initUI(self, frame):
        # Make a grid layout
        layout = pyqt.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # Now we can add widgets to the layout
        f = h5py.File(self.filename, 'r')

        # Get the X and Y coords
        ########################
        R = self.R
        X = R[:, 0]
        Y = R[:, 1]
        self.X = X
        self.Y = Y
        
        self.frames = np.zeros((len(X),), dtype=bool)
        self.frames[self.good_frames] = True
        
        # scatter plot
        ##############
        self.good_frame_pen     = pg.mkPen((255, 150, 150))
        self.good_frame_brush   = pg.mkBrush(255, 255, 255, 120)
        self.bad_frame_pen      = pg.mkPen(None)
        self.selected_frame_brush = pg.mkBrush('b')
        #self.selected_frame_pen = pg.mkPen((150, 150, 255))
        
        self.s1    = pg.ScatterPlotItem(size=5, pen=self.good_frame_pen, brush=self.good_frame_brush)
        spots = [{'pos': [X[i], Y[i]], 'data': i} for i in range(len(R))] 
        self.s1.addPoints(spots)

        # Temp
        ######
        if 'process_3/cpu_stitch/R' in f :
            R = f['process_3/cpu_stitch/R']
            X2 = R[:, 0]
            Y2 = R[:, 1]
            self.X2 = X2
            self.Y2 = Y2
            
            # scatter plot
            ##############
            self.good_frame_pen2     = pg.mkPen((150, 255, 150))
            
            self.s12    = pg.ScatterPlotItem(size=5, pen=self.good_frame_pen2, brush=pg.mkBrush(255, 255, 255, 120))
            spots2 = [{'pos': [X2[i], Y2[i]], 'data': i} for i in range(len(R))] 
            self.s12.addPoints(spots2)
        else :
            self.s12 = None

        ## Make all plots clickable
        def clicked(plot, points):
            for p in points:
                self.frames[p.data()] = ~self.frames[p.data()]
                if self.frames[p.data()] :
                    p.setPen(self.good_frame_pen)
                else :
                    p.setPen(self.bad_frame_pen)

        self.s1.sigClicked.connect(clicked)

        self.update_selected_points()

        ## Show the selected frame
        ##########################
        self.s2     = pg.ScatterPlotItem(size = 10, pxMode=True, brush = self.selected_frame_brush)
        self.s2.setData([self.X[frame]], [self.Y[frame]])

        ## rectangular ROI selection
        ############################
        # put it out of the way
        span    = [0.1 * (X.max()-X.min()), 0.1 * (Y.max()-Y.min())]
        courner = [X.min()-1.5*span[0], Y.min()-1.5*span[1]]
        self.roi = pg.RectROI(courner, span)
        self.roi.setZValue(10)                       # make sure ROI is drawn above image
        ROI_button_good   = pyqt.QPushButton('good frames')
        ROI_button_bad    = pyqt.QPushButton('bad frames')
        ROI_button_toggle = pyqt.QPushButton('toggle frames')
        write_button      = pyqt.QPushButton('write to file')
        ROI_button_good.clicked.connect(   lambda : self.mask_ROI(self.roi, 0))
        ROI_button_bad.clicked.connect(    lambda : self.mask_ROI(self.roi, 1))
        ROI_button_toggle.clicked.connect( lambda : self.mask_ROI(self.roi, 2))
        write_button.clicked.connect(               self.write_good_frames)
        
        scatter_plot = pg.PlotWidget(title='x,y scatter plot', left='y position', bottom='x position')
        scatter_plot.addItem(self.roi)
        scatter_plot.addItem(self.s1)
        scatter_plot.addItem(self.s2)

        if self.s12 is not None :
            scatter_plot.addItem(self.s12)
        
        hbox = pyqt.QHBoxLayout()
        hbox.addWidget(ROI_button_good)
        hbox.addWidget(ROI_button_bad)
        hbox.addWidget(ROI_button_toggle)
        hbox.addWidget(write_button)
        hbox.addStretch(1)
        
        layout.addWidget(scatter_plot   , 0, 0, 1, 1)
        layout.addLayout(hbox           , 1, 0, 1, 1)
        
        f.close()

    def write_good_frames(self):
        f = h5py.File(self.filename, 'a')
        key = 'frame_selector/good_frames'
        if key in f :
            del f[key]
        f[key] = np.where(self.frames)[0]
        f.close()

    def update_selected_points(self):
        pens = [self.good_frame_pen if f else self.bad_frame_pen for f in self.frames]
        self.s1.setPen(pens)

    def replot(self, frame):
        self.s2.setData([self.X[frame]], [self.Y[frame]])

    def mask_ROI(self, roi, good_bad_toggle = 0):
        sides   = [roi.size()[0], roi.size()[1]]
        courner = [roi.pos()[0], roi.pos()[1]]
        
        top_right   = [courner[0] + sides[0], courner[1] + sides[1]]
        bottom_left = courner
        
        y_in_rect = (self.Y <= top_right[1])   & (self.Y >= bottom_left[1])
        x_in_rect = (self.X >= bottom_left[0]) & (self.X <= top_right[0])
        
        if good_bad_toggle == 0 :
            self.frames[ y_in_rect * x_in_rect ] = True
        elif good_bad_toggle == 1 :
            self.frames[ y_in_rect * x_in_rect ] = False
        elif good_bad_toggle == 2 :
            self.frames[ y_in_rect * x_in_rect ] = ~self.frames[ y_in_rect * x_in_rect ]
    
        self.update_selected_points()
