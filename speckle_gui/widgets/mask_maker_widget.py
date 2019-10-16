import numpy as np
import h5py

#import PyQt4.QtGui
try :
    from PyQt5.QtWidgets import *
except :
    from PyQt4.QtGui import *

import pyqtgraph as pg

import speckle_tracking 
from speckle_tracking import config_reader

class Mask_maker_widget(QWidget):
    """
    """
    cspad_psana_shape = (4, 8, 185, 388)
    cspad_geom_shape  = (1480, 1552)
    
    def __init__(self, fnam, data_paths, mask_paths=None, output_file = None, out_path='mask_maker/mask', auto_detect_bitmask=True):
        super(Mask_maker_widget, self).__init__()
        
        f = h5py.File(fnam, 'r')
        
        # load existing mask
        mask = None
        if type(mask_paths) is str and mask_paths is not None and mask_paths in f:
            mask = f[mask_paths][()]
        
        elif mask_paths is not None :
            for mask_path in mask_paths :
                if mask_path is not None and mask_path in f:
                    mask = f[mask_path][()]
                    break
        
        # auto detect bitmask
        if auto_detect_bitmask and mask is not None and np.any(mask > 1):
            # hot (4) and dead (8) pixels
            mask     = ~np.bitwise_and(mask, 4 + 8).astype(np.bool) 
        elif mask is not None :
            mask     = mask.astype(np.bool) 

        # load the data 
        for data_path in data_paths :
            if data_path in f :
                break
        
        # load data frames
        if len(f[data_path].shape) == 2 :
            self.index = None
            cspad = f[data_path][()]
        
        elif len(f[data_path].shape) == 3 :
            self.index = 0
            cspad = f[data_path][0]
        self.fnam      = fnam
        self.data_path = data_path
        
        # set mask if non provided
        if mask is None :
            mask = np.ones_like(cspad).astype(np.bool)
        f.close()
        
        # this is not in fact a cspad image
        self.cspad_shape_flag = 'other'
        self.cspad = cspad

        if output_file is not None :
            self.output_file = output_file
        else :
            self.output_file = fnam

        self.output_path = out_path
        
        self.mask          = np.ones_like(self.cspad, dtype=np.bool)
        self.mask_clicked  = mask
        self.geom_fnam = None
        
        i, j = np.meshgrid(range(self.cspad.shape[0]), range(self.cspad.shape[1]), indexing='ij')
        self.y_map, self.x_map = (i-self.cspad.shape[0]//2, j-self.cspad.shape[1]//2)
        self.cspad_shape = self.cspad.shape
        
        self.mask_edges    = False
        self.mask_unbonded = False

        self.unbonded_pixels = self.make_unbonded_pixels()
        self.asic_edges      = self.make_asic_edges()
        
        self.initUI()
        
    def updateDisplayRGB(self, auto = False):
        """
        Make an RGB image (N, M, 3) (pyqt will interprate this as RGB automatically)
        with masked pixels shown in blue at the maximum value of the cspad. 
        This ensures that the masked pixels are shown at full brightness.
        """
        trans      = np.fliplr(self.cspad.T)
        trans_mask = np.fliplr(self.mask.T)
        self.cspad_max  = self.cspad.max()

        # convert to RGB
        # Set masked pixels to B
        display_data = np.zeros((trans.shape[0], trans.shape[1], 3), dtype = self.cspad.dtype)
        display_data[:, :, 0] = trans * trans_mask
        display_data[:, :, 1] = trans * trans_mask
        display_data[:, :, 2] = trans + (self.cspad_max - trans) * ~trans_mask
        
        self.display_RGB = display_data
        if auto :
            self.plot.setImage(self.display_RGB.astype(np.float))
        else :
            self.plot.setImage(self.display_RGB.astype(np.float), autoRange = False, autoLevels = False, autoHistogramRange = False)

    def generate_mask(self):
        self.mask.fill(1)

        if self.mask_unbonded :
            self.mask *= self.unbonded_pixels

        if self.mask_edges :
            self.mask *= self.asic_edges

        self.mask *= self.mask_clicked

    def update_mask_unbonded(self, state):
        if state > 0 :
            print('adding unbonded pixels to the mask')
            self.mask_unbonded = True
        else :
            print('removing unbonded pixels from the mask')
            self.mask_unbonded = False
        
        self.generate_mask()
        self.updateDisplayRGB()

    def update_mask_edges(self, state):
        if state > 0 :
            print('adding asic edges to the mask')
            self.mask_edges = True
        else :
            print('removing asic edges from the mask')
            self.mask_edges = False
        
        self.generate_mask()
        self.updateDisplayRGB()

    def save_mask(self):
        print('updating mask...')
        self.generate_mask()

        mask = self.mask
        
        print('outputing mask as np.int16 (h5py does not support boolean arrays yet)...')
        f = h5py.File(self.output_file)
        if self.output_path in f :
            del f[self.output_path]
        f[self.output_path] = mask
        f.close()
        print('Done!')
        
    def mask_ROI(self, roi):
        sides   = [roi.size()[1], roi.size()[0]]
        courner = [self.cspad_shape[0]/2. - roi.pos()[1], \
                   roi.pos()[0] - self.cspad_shape[1]/2.]

        top_left     = [np.rint(courner[0]) - 1, np.rint(courner[1])]
        bottom_right = [np.rint(courner[0] - sides[0]), np.rint(courner[1] + sides[1]) - 1]

        y_in_rect = (self.y_map <= top_left[0]) & (self.y_map >= bottom_right[0])
        x_in_rect = (self.x_map >= top_left[1]) & (self.x_map <= bottom_right[1])
        i2, j2 = np.where( y_in_rect & x_in_rect )
        self.apply_ROI(i2, j2)

    def mask_ROI_circle(self, roi):
        # get the xy coords of the centre and the radius
        rad    = roi.size()[0]/2. + 0.5
        centre = [self.cspad_shape[0]/2. - roi.pos()[1] - rad, \
                  roi.pos()[0] + rad - self.cspad_shape[1]/2.]
        
        r_map = np.sqrt((self.y_map-centre[0])**2 + (self.x_map-centre[1])**2)
        i2, j2 = np.where( r_map <= rad )
        self.apply_ROI(i2, j2)

    def apply_ROI(self, i2, j2):
        if self.toggle_checkbox.isChecked():
            self.mask_clicked[i2, j2] = ~self.mask_clicked[i2, j2]
        elif self.mask_checkbox.isChecked():
            self.mask_clicked[i2, j2] = False
        elif self.unmask_checkbox.isChecked():
            self.mask_clicked[i2, j2] = True
        
        self.generate_mask()
        self.updateDisplayRGB()
    
    def mask_hist(self):
        min_max = self.plot.getHistogramWidget().item.getLevels()
        
        if self.toggle_checkbox.isChecked():
            self.mask_clicked[np.where(self.cspad < min_max[0])] = ~self.mask_clicked[np.where(self.cspad < min_max[0])]
            self.mask_clicked[np.where(self.cspad > min_max[1])] = ~self.mask_clicked[np.where(self.cspad > min_max[1])]
        elif self.mask_checkbox.isChecked():
            self.mask_clicked[np.where(self.cspad < min_max[0])] = False
            self.mask_clicked[np.where(self.cspad > min_max[1])] = False
        elif self.unmask_checkbox.isChecked():
            self.mask_clicked[np.where(self.cspad < min_max[0])] = True
            self.mask_clicked[np.where(self.cspad > min_max[1])] = True
        
        self.generate_mask()
        self.updateDisplayRGB()

    def initUI(self):
        ## 2D plot for the cspad and mask
        #################################
        self.plot = pg.ImageView()

        ## save mask button
        #################################
        save_button = QPushButton('save mask')
        save_button.clicked.connect(self.save_mask)

        # rectangular ROI selection
        #################################
        self.roi = pg.RectROI([-200,-200], [100, 100])
        self.plot.addItem(self.roi)
        self.roi.setZValue(10)                       # make sure ROI is drawn above image
        ROI_button = QPushButton('mask rectangular ROI')
        ROI_button.clicked.connect(lambda : self.mask_ROI(self.roi))

        # circular ROI selection
        #################################
        self.roi_circle = pg.CircleROI([-200,200], [101, 101])
        self.plot.addItem(self.roi_circle)
        self.roi.setZValue(10)                       # make sure ROI is drawn above image
        ROI_circle_button = QPushButton('mask circular ROI')
        ROI_circle_button.clicked.connect(lambda : self.mask_ROI_circle(self.roi_circle))

        # histogram mask button
        #################################
        hist_button = QPushButton('mask outside histogram')
        hist_button.clicked.connect(self.mask_hist)

        # prev / next buttons
        #################################
        hbox = QHBoxLayout()
        prev_button = QPushButton('prev frame')
        prev_button.clicked.connect(self.prev_frame)
        next_button = QPushButton('next frame')
        next_button.clicked.connect(self.next_frame)
        hbox.addWidget(prev_button)
        hbox.addWidget(next_button)
        if self.index is None :
            next_button.setEnabled(False)
            prev_button.setEnabled(False)
        
        # toggle / mask / unmask checkboxes
        #################################
        self.toggle_checkbox   = QCheckBox('toggle')
        self.mask_checkbox     = QCheckBox('mask')
        self.unmask_checkbox   = QCheckBox('unmask')
        self.toggle_checkbox.setChecked(True)   
        
        self.toggle_group      = QButtonGroup()#"masking behaviour")
        self.toggle_group.addButton(self.toggle_checkbox)   
        self.toggle_group.addButton(self.mask_checkbox)   
        self.toggle_group.addButton(self.unmask_checkbox)   
        self.toggle_group.setExclusive(True)
        
        # toggle / mask / unmask checkboxes
        #################################
        self.toggle_checkbox   = QCheckBox('toggle')
        self.mask_checkbox     = QCheckBox('mask')
        self.unmask_checkbox   = QCheckBox('unmask')
        self.toggle_checkbox.setChecked(True)   
        
        self.toggle_group      = QButtonGroup()#"masking behaviour")
        self.toggle_group.addButton(self.toggle_checkbox)   
        self.toggle_group.addButton(self.mask_checkbox)   
        self.toggle_group.addButton(self.unmask_checkbox)   
        self.toggle_group.setExclusive(True)

        # mouse hover ij value label
        #################################
        ij_label = QLabel()
        disp = 'ss fs {0:5} {1:5}   value {2:2}'.format('-', '-', '-')
        ij_label.setText(disp)
        self.plot.scene.sigMouseMoved.connect( lambda pos: self.mouseMoved(ij_label, pos) )
        
        # unbonded pixels checkbox
        #################################
        unbonded_checkbox = QCheckBox('unbonded pixels')
        unbonded_checkbox.stateChanged.connect( self.update_mask_unbonded )
        if self.cspad_shape_flag == 'other' :
            unbonded_checkbox.setEnabled(False)
        
        # asic edges checkbox
        #################################
        edges_checkbox = QCheckBox('asic edges')
        edges_checkbox.stateChanged.connect( self.update_mask_edges )
        if self.cspad_shape_flag == 'other' :
            edges_checkbox.setEnabled(False)
        
        # mouse click mask 
        #################################
        self.plot.scene.sigMouseClicked.connect( lambda click: self.mouseClicked(self.plot, click) )

        # Create a grid layout to manage the widgets size and position
        #################################
        layout = QGridLayout()
        self.setLayout(layout)

        ## Add widgets to the layout in their proper positions
        layout.addWidget(save_button, 0, 0)             # upper-left
        layout.addWidget(ROI_button, 1, 0)              # upper-left
        layout.addWidget(ROI_circle_button, 2, 0)       # upper-left
        layout.addWidget(hist_button, 3, 0)             # upper-left
        layout.addLayout(hbox, 4, 0)             # upper-left
        layout.addWidget(self.toggle_checkbox, 5, 0)    # upper-left
        layout.addWidget(self.mask_checkbox, 6, 0)      # upper-left
        layout.addWidget(self.unmask_checkbox, 7, 0)    # upper-left
        layout.addWidget(ij_label, 8, 0)                # upper-left
        layout.addWidget(unbonded_checkbox, 9, 0)       # middle-left
        layout.addWidget(edges_checkbox, 10, 0)          # bottom-left
        layout.addWidget(self.plot, 0, 1, 10, 1)         # plot goes on right side, spanning 3 rows
        layout.setColumnStretch(1, 1)
        layout.setColumnMinimumWidth(0, 250)
        
        # display the image
        self.generate_mask()
        self.updateDisplayRGB(auto = True)

    def prev_frame(self):
        if self.index > 0 :
            self.index -= 1
            f = h5py.File(self.fnam, 'r')
            self.cspad = f[self.data_path][self.index]
            f.close()
            self.updateDisplayRGB()

    def next_frame(self):
        f = h5py.File(self.fnam, 'r')
        if self.index < (f[self.data_path].shape[0]-1):
            self.index += 1
            self.cspad  = f[self.data_path][self.index]
            self.updateDisplayRGB()
        f.close()

    def mouseMoved(self, ij_label, pos):
        img = self.plot.getImageItem()
        if self.geom_fnam is not None :
            ij = [self.cspad_shape[0] - 1 - int(img.mapFromScene(pos).y()), int(img.mapFromScene(pos).x())] # ss, fs
            if (0 <= ij[0] < self.cspad_shape[0]) and (0 <= ij[1] < self.cspad_shape[1]):
                ij_label.setText('ss fs value: %d %d %.2e' % (self.ss_geom[ij[0], ij[1]], self.fs_geom[ij[0], ij[1]], self.cspad_geom[ij[0], ij[1]]) )
        else :
            ij = [self.cspad.shape[0] - 1 - int(img.mapFromScene(pos).y()), int(img.mapFromScene(pos).x())] # ss, fs
            if (0 <= ij[0] < self.cspad.shape[0]) and (0 <= ij[1] < self.cspad.shape[1]):
                ij_label.setText('ss fs value: %d %d %.2e' % (ij[0], ij[1], self.cspad[ij[0], ij[1]]) )

    def mouseClicked(self, plot, click):
        if click.button() == 1:
            img = plot.getImageItem()
            i0 = int(img.mapFromScene(click.pos()).y())
            j0 = int(img.mapFromScene(click.pos()).x())
            i1 = self.cspad.shape[0] - 1 - i0 # array ss (with the fliplr and .T)
            j1 = j0                           # array fs (with the fliplr and .T)
            if (0 <= i1 < self.cspad.shape[0]) and (0 <= j1 < self.cspad.shape[1]):
                if self.toggle_checkbox.isChecked():
                    self.mask_clicked[i1, j1] = ~self.mask_clicked[i1, j1]
                    self.mask[i1, j1]         = ~self.mask[i1, j1]
                elif self.mask_checkbox.isChecked():
                    self.mask_clicked[i1, j1] = False
                    self.mask[i1, j1]         = False
                elif self.unmask_checkbox.isChecked():
                    self.mask_clicked[i1, j1] = True
                    self.mask[i1, j1]         = True
                if self.mask[i1, j1] :
                    self.display_RGB[j0, i0, :] = np.array([1,1,1]) * self.cspad[i1, j1]
                else :
                    self.display_RGB[j0, i0, :] = np.array([0,0,1]) * self.cspad_max
            
            self.plot.setImage(self.display_RGB.astype(np.float), autoRange = False, autoLevels = False, autoHistogramRange = False)
    
    def make_unbonded_pixels(self):
        cspad_psana_shape = self.cspad_psana_shape
        cspad_geom_shape  = self.cspad_geom_shape

        def ijkl_to_ss_fs(cspad_ijkl):
            """ 
            0: 388        388: 2 * 388  2*388: 3*388  3*388: 4*388
            (0, 0, :, :)  (1, 0, :, :)  (2, 0, :, :)  (3, 0, :, :)
            (0, 1, :, :)  (1, 1, :, :)  (2, 1, :, :)  (3, 1, :, :)
            (0, 2, :, :)  (1, 2, :, :)  (2, 2, :, :)  (3, 2, :, :)
            ...           ...           ...           ...
            (0, 7, :, :)  (1, 7, :, :)  (2, 7, :, :)  (3, 7, :, :)
            """
            if cspad_ijkl.shape != cspad_psana_shape :
                raise ValueError('cspad input is not the required shape:' + str(cspad_psana_shape) )

            cspad_ij = np.zeros(cspad_geom_shape, dtype=cspad_ijkl.dtype)
            for i in range(4):
                cspad_ij[:, i * cspad_psana_shape[3]: (i+1) * cspad_psana_shape[3]] = cspad_ijkl[i].reshape((cspad_psana_shape[1] * cspad_psana_shape[2], cspad_psana_shape[3]))

            return cspad_ij

        mask = np.ones(cspad_psana_shape)

        for q in range(cspad_psana_shape[0]):
            for p in range(cspad_psana_shape[1]):
                for a in range(2):
                    for i in range(19):
                        mask[q, p, i * 10, i * 10] = 0
                        mask[q, p, i * 10, i * 10 + cspad_psana_shape[-1]//2] = 0

        mask_slab = ijkl_to_ss_fs(mask)

        import scipy.signal
        kernal = np.array([ [0,1,0], [1,1,1], [0,1,0] ], dtype=np.float)
        mask_pad = scipy.signal.convolve(1 - mask_slab.astype(np.float), kernal, mode = 'same') < 1
        return mask_pad

    def make_asic_edges(self, arrayin = None, pad = 0):
        mask_edges = np.ones(self.cspad_geom_shape, dtype=np.bool)
        mask_edges[:: 185, :] = 0
        mask_edges[184 :: 185, :] = 0
        mask_edges[:, :: 194] = 0
        mask_edges[:, 193 :: 194] = 0

        if pad != 0 :
            mask_edges = scipy.signal.convolve(1 - mask_edges.astype(np.float), np.ones((pad, pad), dtype=np.float), mode = 'same') < 1
        return mask_edges

    def edges(self, shape, pad = 0):
        mask_edges = np.ones(shape)
        mask_edges[0, :]  = 0
        mask_edges[-1, :] = 0
        mask_edges[:, 0]  = 0
        mask_edges[:, -1] = 0

        if pad != 0 :
            mask_edges = scipy.signal.convolve(1 - mask_edges.astype(np.float), np.ones((pad, pad), dtype=np.float), mode = 'same') < 1
        return mask_edges
