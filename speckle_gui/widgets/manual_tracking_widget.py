# make a gui that shows 
# Atlas | frame

# then click (alternating) on the atlas and the gui

import numpy as np
import h5py

#import PyQt4.QtGui
try :
    from PyQt5.QtWidgets import *
except :
    from PyQt4.QtGui import *

import pyqtgraph as pg

import sys, os
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
root = os.path.split(root)[0]

from config_editor_widget import Config_editor_Widget
from config_editor_widget import discover_config
from run_and_log_command  import Run_and_log_command
import config_reader

class Manual_tracking_widget(QWidget):
    script_name = 'manual_tracking'
    
    def __init__(self, h5_fnam, config_fnam = '', config_dirs = ['/process/',]):
        super(Manual_tracking_widget, self).__init__()
        
        self.config_fnams, self.config_output =  discover_config(self.script_name, \
                                                 h5_fnam, config_fnam, config_dirs)
        self.filename    = h5_fnam
        self.initUI()

    def initUI(self):
        # Make a grid layout
        ####################
        layout = QGridLayout()
        self.setLayout(layout)
        
        # make a vbox for the left column
        vst = QVBoxLayout()
        
        # config widget
        ###############
        self.cew = Config_editor_Widget(self.config_fnams, self.config_output)
        self.config, fnam = config_reader.config_read(self.config_fnams)
        self.config       = self.config[self.script_name]
        vst.addWidget(self.cew)
        
        # get feature list
        ##################
        # dict of dict of lists
        self.features = self.read_features()
        
        # show frames with scatter plot
        ###############################
        self.init_display()
        
        # add feature grid button
        #########################
        #hst = QHBoxLayout()
        #add_feature_grid_button   = QPushButton('add feature grid')
        #add_feature_grid_button.clicked.connect( self.add_feature_grid )
        #self.add_feature_grid_lineedit = QLineEdit()
        #self.add_feature_grid_lineedit.setText('4')
        #hst.addWidget(add_feature_grid_button)
        #hst.addWidget(self.add_feature_grid_lineedit)
        #vst.addLayout(hst)

        # add feature button
        ####################
        add_feature_button    = QPushButton('add feature')
        add_feature_button.clicked.connect( self.add_feature )
        vst.addWidget(add_feature_button)

        # remove feature button
        #######################
        remove_feature_button = QPushButton('remove feature')
        remove_feature_button.clicked.connect( self.remove_feature )
        vst.addWidget(remove_feature_button)
        
        # clear all button
        ##################
        clear_all_button = QPushButton('clear all features')
        clear_all_button.clicked.connect( self.remove_all_features )
        vst.addWidget(clear_all_button)
        
        # write feature button
        ######################
        write_features_button = QPushButton('write features')
        write_features_button.clicked.connect( self.write_features )
        vst.addWidget(write_features_button)

        # merge feature button
        ######################
        #merge_features_button = QPushButton('merge features')
        #merge_features_button.clicked.connect( self.merge_features )
        #vst.addWidget(merge_features_button)
        
        # next duplicate button
        ##################
        next_pair_button = QPushButton('next and duplicate')
        next_pair_button.clicked.connect( self.next_pair )
        vst.addWidget(next_pair_button)

        # next pair button
        ##################
        hst = QHBoxLayout()
        nextp_button = QPushButton('next pair')
        nextp_button.clicked.connect( self.nextp )
        prevp_button = QPushButton('previous pair')
        prevp_button.clicked.connect( self.prevp )
        hst.addWidget(prevp_button)
        hst.addWidget(nextp_button)
        vst.addLayout(hst)

        # run command widget
        ####################
        self.run_command_widget = Run_and_log_command()
        self.run_command_widget.finished_signal.connect(self.finished)
        
        # run command button
        ####################
        self.run_button = QPushButton(self.script_name, self)
        self.run_button.clicked.connect(self.run_button_clicked)
        vst.addWidget(self.run_button)

        # run command button
        ####################
        self.run2_button = QPushButton('add distortions', self)
        self.run2_button.clicked.connect(self.run2_button_clicked)
        vst.addWidget(self.run2_button)
        
        # add a spacer for the labels and such
        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        vst.addItem(verticalSpacer)
        
        # set the layout
        ################
        layout.addLayout(vst              ,       0, 0, 1, 1)
        layout.addLayout(self.imgs_layout ,       0, 1, 1, 1)
        layout.addWidget(self.run_command_widget , 1, 0, 1, 2)
        
        #layout.addItem(verticalSpacer,             1, 0, 1, 1)
        layout.setColumnStretch(1, 1)
        #layout.setColumnMinimumWidth(0, 250)
        self.layout = layout

    def nextp(self):
        """
        jump to the next frame pair with features
        ordering: o = frames * i + j
        """
        i = self.frame1_index
        for j in range(self.frame2_index+1, self.N, 1):
            # is there a match?
            if i in self.features and j in self.features[i] :
                self.update_display(i, j)
                return
        
        for i in range(self.frame1_index+1, self.N, 1):
            for j in range(0, self.N, 1):
                # is there a match?
                if i in self.features and j in self.features[i] :
                    self.update_display(i, j)
                    return

    
    def prevp(self):
        i = self.frame1_index
        for j in range(self.frame2_index-1, -1, -1):
            # is there a match?
            if i in self.features and j in self.features[i] :
                self.update_display(i, j)
                return
        
        for i in range(self.frame1_index-1, -1, -1):
            for j in range(self.N-1, -1, -1):
                # is there a match?
                if i in self.features and j in self.features[i] :
                    self.update_display(i, j)
                    return

    def next_pair(self):
        """
        increment the frame indices
        if there are no features for this pair then copy them from the last pair
        """
        any_features_current = False
        if self.frame1_index in self.features :
            if self.frame2_index in self.features[self.frame1_index] :
                any_features_current = True
                
                pos = []
                for fe in self.features[self.frame1_index][self.frame2_index]:
                    pos.append([[fe[-2].pos()[0],fe[-2].pos()[1]], [fe[-1].pos()[0],fe[-1].pos()[1]]])
        
        frame1_index = self.frame1_index + 1
        frame2_index = self.frame2_index + 1
        
        if frame1_index >= self.N or self.frame2_index >= self.N :
            print('end of file...')
            return
        
        any_features_next = False
        if frame1_index in self.features :
            if frame2_index in self.features[frame1_index] :
                any_features_next = True
        
        if any_features_current is True and any_features_next is False :
            w = [self.config['window'], self.config['window']]
            
            for p in pos:
                # add an roi to frame1 and frame2
                if frame1_index not in self.features :
                    self.features[frame1_index] = {frame2_index : []}

                if frame2_index not in self.features[frame1_index] :
                    self.features[frame1_index][frame2_index] = []
                
                pen = self.get_pen()
                self.features[frame1_index][frame2_index].append(\
                        [pg.RectROI([int(p[0][0]),int(p[0][1])], w, translateSnap=True, pen = pen), \
                         pg.RectROI([int(p[1][0]),int(p[1][1])], w, translateSnap=True, pen = pen)])
        
        self.update_display(frame1_index, frame2_index)
        
    def remove_all_features(self):
        # remove the last roi's 
        self.remove_features_from_screen()
        self.features = {}
    
    def run_button_clicked(self):
        # Run the command 
        #################
        py = 'process/manual_tracking.py'
        cmd = 'mpiexec -n 8 python ' + py + ' ' + self.filename + ' -c ' + self.config_output
        self.run_command_widget.run_cmd(cmd)

    def run2_button_clicked(self):
        # Run the command 
        #################
        py = 'utils/add_distortions.py'
        cmd = 'python ' + py + ' ' + self.filename + ' -c ' + self.config_output
        self.run_command_widget.run_cmd(cmd)

    def finished(self):
        self.remove_features_from_screen()
        self.features = self.read_features()
        self.update_display(self.frame1_index, self.frame2_index, update_atlas = True)

    def init_display(self):
        # get file info
        roi    = self.config['roi']
        # show a frame
        self.f = h5py.File(self.filename, 'r')
        self.mask = self.f[self.config['mask']][roi[0]:roi[1], roi[2]:roi[3]]
        self.W    = self.f[self.config['whitefield']][roi[0]:roi[1], roi[2]:roi[3]]*self.mask
        self.W[self.W==0] = 1
        self.N      = self.f[self.config['frames']].shape[0]
        self.f.close()
        
        self.frame1_index = -1
        self.frame2_index = -1
        
        # atlas
        atlas_plt = pg.PlotItem(title = 'atlas')
        self.atlas_imageView = pg.ImageView(view = atlas_plt)
        self.atlas_imageView.ui.menuBtn.hide()
        self.atlas_imageView.ui.roiBtn.hide()
        
        # frame1
        self.frame1_plt = pg.PlotItem(title = 'Frame 1')
        self.frame1_imageView = pg.ImageView(view = self.frame1_plt)
        self.frame1_imageView.ui.menuBtn.hide()
        self.frame1_imageView.ui.roiBtn.hide()
        
        # frame2
        self.frame2_plt = pg.PlotItem(title = 'Frame 2')
        self.frame2_imageView = pg.ImageView(view = self.frame2_plt)
        self.frame2_imageView.ui.menuBtn.hide()
        self.frame2_imageView.ui.roiBtn.hide()
        
        # frame1 vline
        self.frame1_position_plotsW = pg.PlotWidget()
        self.frame1_position_plotsW.plot(np.arange(self.N))
        self.frame1_vline = self.frame1_position_plotsW.addLine(x = 0, movable=True, bounds = [0, self.N-1])
        self.frame1_position_plotsW.setFixedHeight(50)
        
        # frame2 vline
        self.frame2_position_plotsW = pg.PlotWidget()
        self.frame2_position_plotsW.plot(np.arange(self.N))
        self.frame2_vline = self.frame2_position_plotsW.addLine(x = 1, movable=True, bounds = [0, self.N-1])
        self.frame2_position_plotsW.setFixedHeight(50)
        
        self.frame1_vline.sigPositionChanged.connect(lambda x : self.update_display(int(self.frame1_vline.value()), self.frame2_index))
        self.frame2_vline.sigPositionChanged.connect(lambda x : self.update_display(self.frame1_index, int(self.frame2_vline.value())))

        # make the layout
        self.imgs_layout = QGridLayout()

        vst1 = QVBoxLayout()
        vst1.addWidget(self.frame1_imageView)
        vst1.addWidget(self.frame1_position_plotsW)

        vst2 = QVBoxLayout()
        vst2.addWidget(self.frame2_imageView)
        vst2.addWidget(self.frame2_position_plotsW)
        
        self.imgs_layout.addWidget(self.atlas_imageView,     0, 0, 1, 2)
        self.imgs_layout.addLayout(vst1,                     1, 0, 1, 1)
        self.imgs_layout.addLayout(vst2,                     1, 1, 1, 1)
        self.imgs_layout.setColumnStretch(0, 1)
        self.imgs_layout.setColumnStretch(1, 1)
        self.update_display(1, 2, update_atlas=True)

    def get_pen(self):
        pen = np.random.randint(0, 256, 3)

        # make sure it's bright
        pen = np.rint(255 * pen / pen.max().astype(np.float)).astype(np.int)

        # make sure it's not white
        if np.std(pen) < 10. :
            pen[np.random.randint(0, 3, 1)] = 0
        return pg.mkPen(pen)

    def add_feature_grid(self):
        roi    = self.config['roi']
        w = [self.config['window'], self.config['window']]
        # get the shift between frames
        f = h5py.File(self.filename)
        if (self.config['h5_group']+'/pix_positions') in f :
            pos = f[self.config['h5_group']+'/pix_positions'][()]
            f.close()
            p1 = None
            p2 = None
            for p in pos :
                print(p)
                if p[0] == self.frame1_index :
                    p1 = p[1:]
                elif p[0] == self.frame2_index :
                    p2 = p[1:]
            print('positions:', p1, p2)
            if p1 is None or p2 is None :
                print('no positions found for frames:', self.frame1_index, self.frame2_index)
                return
        else :
            print('no positions found...')
            f.close()
            return

        shift = p2-p1
        N, M  = roi[1]-roi[0], roi[3] - roi[2]
        i1_min, i1_max = max((1+w[0])//2, -shift[0]), min((1-w[0])//2 + N, N - shift[0])
        j1_min, j1_max = max((1+w[1])//2, -shift[1]), min((1-w[1])//2 + M, M - shift[1])
        i2_min, i2_max = max((1+w[0])//2, shift[0]), min((1-w[0])//2 + N, N + shift[0])
        j2_min, j2_max = max((1+w[1])//2, shift[1]), min((1-w[1])//2 + M, M + shift[1])

        N      = int(self.add_feature_grid_lineedit.text())
        i1g, j1g = np.linspace(i1_min, i1_max, N).astype(np.int), \
                   np.linspace(j1_min, j1_max, N).astype(np.int)
        i2g, j2g = np.linspace(i2_min, i2_max, N).astype(np.int), \
                   np.linspace(j2_min, j2_max, N).astype(np.int)
        i1g, j1g = np.meshgrid(i1g, j1g, indexing='ij')
        i2g, j2g = np.meshgrid(i2g, j2g, indexing='ij')
        for i1, j1, i2, j2 in zip(i1g.ravel(), j1g.ravel(), i2g.ravel(), j2g.ravel()):
            print(i1, j1, i2, j2)
            self.add_feature(i1, j1, i2, j2)

    def add_feature(self, i1=0, j1=0, i2=0, j2=0):
        w = [self.config['window'], self.config['window']]
        
        pen = self.get_pen()
        fe1 = pg.RectROI([i1 + ((1-w[0])//2),j1 + ((1-w[0])//2)], w, translateSnap=True, pen = pen)
        fe2 = pg.RectROI([i2 + ((1-w[0])//2),j2 + ((1-w[0])//2)], w, translateSnap=True, pen = pen)
        
        self.frame1_imageView.addItem(fe1)
        self.frame2_imageView.addItem(fe2)
        
        fe1.removeHandle(0)
        fe1.setZValue(10) 
        fe2.removeHandle(0)
        fe2.setZValue(10)
        
        # add an roi to frame1 and frame2
        if self.frame1_index not in self.features :
            self.features[self.frame1_index] = {self.frame2_index : []}

        if self.frame2_index not in self.features[self.frame1_index] :
            self.features[self.frame1_index][self.frame2_index] = []
        
        self.features[self.frame1_index][self.frame2_index].append([fe1, fe2])
        
    def remove_feature(self):
        if self.frame1_index in self.features :
            if self.frame2_index in self.features[self.frame1_index] :
                fe1, fe2 = self.features[self.frame1_index][self.frame2_index][-1][-2:]
                
                self.frame1_imageView.removeItem(fe1)
                self.frame2_imageView.removeItem(fe2)
                
                self.features[self.frame1_index][self.frame2_index].pop(-1)
                
                # remove stuff
                if len(self.features[self.frame1_index][self.frame2_index]) == 0 :
                    self.features[self.frame1_index].pop(self.frame2_index)
                    if len(self.features[self.frame1_index]) == 0 :
                        self.features.pop(self.frame1_index)

    def write_features(self):
        roi    = self.config['roi']
        w      = [self.config['window'], self.config['window']]
        
        print('outputing features:')
        out = []
        for k1 in self.features.keys():
            for k2 in self.features[k1].keys():
                if k1 >= self.N or k2 >= self.N :
                    continue 
                
                for fe in self.features[k1][k2]:
                    fe1, fe2 = fe[-2:]
                    # carefull with the //'s !!!
                    i1, j1 = fe1.pos()[0]-((1-w[0])//2)+roi[0], fe1.pos()[1]-((1-w[1])//2)+roi[2]
                    i2, j2 = fe2.pos()[0]-((1-w[0])//2)+roi[0], fe2.pos()[1]-((1-w[1])//2)+roi[2]
                    out.append(np.rint(np.array([k1, i1, j1, k2, i2, j2])).astype(np.int))
                    print(out[-1])
        
        # apply offset for the roi and window size
        out = np.array(out) 

        if len(out) > 0 :
            f = h5py.File(self.filename)
            if self.config['features'] in f :
                del f[self.config['features']]
            f[self.config['features']] = out
            f.close()
            print('\nfeatures written')
        else :
            print('\nno features to write')

    def read_features(self):
        config = self.config
        roi    = config['roi']
        w      = [config['window'], config['window']]
        
        features = {}
        f = h5py.File(self.filename, 'r')
        if config['features'] in f :
            fes = np.rint(f[config['features']][()]).astype(np.int)
            f.close()
            
            for fe in fes :
                if fe[0] not in features :
                    features[fe[0]] = {fe[3] : []}
                
                if fe[3] not in features[fe[0]] :
                    features[fe[0]][fe[3]] = []
                
                pos1 = (fe[1] + (1-w[0])//2-roi[0], fe[2] + (1-w[1])//2-roi[2])
                pos2 = (fe[4] + (1-w[0])//2-roi[0], fe[5] + (1-w[1])//2-roi[2])
                    
                pen = self.get_pen()
                features[fe[0]][fe[3]].append([pg.RectROI(pos1, w, translateSnap=True, pen = pen), 
                                               pg.RectROI(pos2, w, translateSnap=True, pen = pen)])
        return features

    def remove_features_from_screen(self):
        # remove the last roi's 
        if self.frame1_index in self.features :
            if self.frame2_index in self.features[self.frame1_index] :
                for fe in self.features[self.frame1_index][self.frame2_index]:
                    print('removing roi...')
                    fe1, fe2 = fe[-2:]
                    self.frame1_imageView.removeItem(fe1)
                    self.frame2_imageView.removeItem(fe2)

    def update_display(self, i, j, update_atlas = False):
        config = self.config
        roi    = config['roi']
        
        if i >= self.N or j >= self.N :
            print('end of file...')
            return 

        f = h5py.File(self.filename, 'r')
        if i != self.frame1_index :
            frame1 = f[config['frames']][i][roi[0]:roi[1], roi[2]:roi[3]]*self.mask / self.W.astype(np.float)
            self.frame1_imageView.setImage(frame1, autoRange = False, autoLevels = False, autoHistogramRange = False)
            self.frame1_plt.setTitle('Frame '+ str(i))
        
        if j != self.frame2_index :
            frame2 = f[config['frames']][j][roi[0]:roi[1], roi[2]:roi[3]]*self.mask / self.W.astype(np.float)
            self.frame2_imageView.setImage(frame2, autoRange = False, autoLevels = False, autoHistogramRange = False)
            self.frame2_plt.setTitle('Frame '+ str(j))

        if update_atlas :
            print('updating atlas', )
            if config['atlas'] in f :
                atlas  = f[config['atlas']][()]
                self.atlas_imageView.setImage(atlas, autoRange = False, autoLevels = False, autoHistogramRange = False)
            else :
                print('atlas not in file')
                atlas  = None
        f.close()
        
        # remove the last roi's 
        self.remove_features_from_screen()
        
        self.frame1_index = i
        self.frame2_index = j
        
        # add the new roi's 
        if self.frame1_index in self.features :
            if self.frame2_index in self.features[self.frame1_index] :
                for fe in self.features[self.frame1_index][self.frame2_index]:
                    fe1, fe2 = fe[-2:]
                    
                    self.frame1_imageView.addItem(fe1)
                    self.frame2_imageView.addItem(fe2)
                    
                    try :
                        fe1.removeHandle(0)
                    except IndexError :
                        pass
                    try :
                        fe2.removeHandle(0)
                    except IndexError :
                        pass

                    fe1.setZValue(10) 
                    fe2.setZValue(10)

                
