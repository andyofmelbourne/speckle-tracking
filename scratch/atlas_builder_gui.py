# make a gui that shows 
# Atlas | frame

# then click (alternating) on the atlas and the gui

import sys, os
import numpy as np
import h5py

#import PyQt4.QtGui
try :
    from PyQt5 import QtGui, QtCore
except :
    from PyQt4 import QtGui, QtCore
import signal
import copy 
import pyqtgraph as pg
import utils


try :
    import configparser
except ImportError :
    import ConfigParser as configparser


def load_config(filename, name):
    # if config is None then read the default from the *.pty dir
    config = os.path.join(os.path.split(filename)[0], name)
    if not os.path.exists(config):
        config = os.path.join(root, 'process')
        config = os.path.join(config, name)
    
    # check that args.config exists
    if not os.path.exists(config):
        raise NameError('config file does not exist: ' + config)
    
    # process config file
    conf = configparser.ConfigParser()
    conf.read(config)
    
    params = utils.parse_parameters(conf)
    return params

class Write_config_file_widget(QtGui.QWidget):
    def __init__(self, config_dict, output_filename):
        super(Write_config_file_widget, self).__init__()
        
        self.output_filename = output_filename
        self.initUI(config_dict)
    
    def initUI(self, config_dict):
        # Make a grid layout
        layout = QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        self.output_config = copy.deepcopy(config_dict)
        
        i = 0
        # add the output config filename 
        ################################    
        fnam_label = QtGui.QLabel(self)
        fnam_label.setText(self.output_filename)
        
        # add the label to the layout
        layout.addWidget(fnam_label, i, 0, 1, 2)
        i += 1
        
        # we have 
        self.labels_lineedits = {}
        group_labels = []
        for group in config_dict.keys():
            # add a label for the group
            group_labels.append( QtGui.QLabel(self) )
            group_labels[-1].setText(group)
            # add the label to the layout
            layout.addWidget(group_labels[-1], i, 0, 1, 2)
            i += 1
            
            self.labels_lineedits[group] = {}
            # add the labels and line edits
            for key in config_dict[group].keys():
                self.labels_lineedits[group][key] = {}
                self.labels_lineedits[group][key]['label'] = QtGui.QLabel(self)
                self.labels_lineedits[group][key]['label'].setText(key)
                layout.addWidget(self.labels_lineedits[group][key]['label'], i, 0, 1, 1)
                
                self.labels_lineedits[group][key]['lineedit'] = QtGui.QLineEdit(self)
                # special case when config_dict[group][key] is a list
                if type(config_dict[group][key]) is list or type(config_dict[group][key]) is np.ndarray :
                    setT = ''
                    for ii in range(len(config_dict[group][key])-1):
                        setT += str(config_dict[group][key][ii]) + ','
                    setT += str(config_dict[group][key][-1])
                else :
                    setT = str(config_dict[group][key])
                self.labels_lineedits[group][key]['lineedit'].setText(setT)
                self.labels_lineedits[group][key]['lineedit'].editingFinished.connect(self.write_file)
                layout.addWidget(self.labels_lineedits[group][key]['lineedit'], i, 1, 1, 1)
                i += 1

    def write_file(self):
        with open(self.output_filename, 'w') as f:
            for group in self.labels_lineedits.keys():
                f.write('['+group+']' + '\n')
                
                for key in self.labels_lineedits[group].keys():
                    out_str = key
                    out_str = out_str + ' = '
                    out_str = out_str + str(self.labels_lineedits[group][key]['lineedit'].text())
                    f.write( out_str + '\n')

class Run_and_log_command(QtGui.QWidget):
    """
    run a command and send a signal when it complete, or it has failed.

    use a Qt timer to check the process
    
    realtime streaming of the terminal output has so proved to be fruitless
    """
    finished_signal = QtCore.pyqtSignal(bool)
    
    def __init__(self):
        super(Run_and_log_command, self).__init__()
        
        self.polling_interval = 0.1
        self.initUI()
        
    def initUI(self):
        """
        Just setup a qlabel showing the shell command
        and another showing the status of the process
        """
        # Make a grid layout
        #layout = QtGui.QGridLayout()
        hbox = QtGui.QHBoxLayout()
        
        # add the layout to the central widget
        self.setLayout(hbox)
        
        # show the command being executed
        self.command_label0 = QtGui.QLabel(self)
        self.command_label0.setText('<b>Command:</b>')
        self.command_label  = QtGui.QLabel(self)
        #self.command_label.setMaximumSize(50, 250)
         
        # show the status of the command
        self.status_label0  = QtGui.QLabel(self)
        self.status_label0.setText('<b>Status:</b>')
        self.status_label   = QtGui.QLabel(self)
        
        # add to layout
        hbox.addWidget(self.status_label0)
        hbox.addWidget(self.status_label)
        hbox.addWidget(self.command_label0)
        hbox.addWidget(self.command_label)
        hbox.addStretch(1)

        #layout.addWidget(self.status_label0,  0, 0, 1, 1)
        #layout.addWidget(self.status_label,   0, 1, 1, 1)
        #layout.addWidget(self.command_label0, 1, 0, 1, 1)
        #layout.addWidget(self.command_label,  1, 1, 1, 1)
         
    def run_cmd(self, cmd):
        from subprocess import PIPE, Popen
        import shlex
        self.command_label.setText(cmd)
        self.status_label.setText('running the command')
        self.p = Popen(shlex.split(cmd), stdout = PIPE, stderr = PIPE)
        
        # start a Qt timer to update the status
        QtCore.QTimer.singleShot(self.polling_interval, self.update_status)
    
    def update_status(self):
        status = self.p.poll()
        if status is None :
            self.status_label.setText('Running')
             
            # start a Qt timer to update the status
            QtCore.QTimer.singleShot(self.polling_interval, self.update_status)
        
        elif status is 0 :
            self.status_label.setText('Finished')
            
            # get the output and error msg
            self.output, self.err_msg = self.p.communicate()
            
            # emmit a signal when complete
            self.finished_signal.emit(True)
            print('Output   :', self.output.decode("utf-8"))
            
        else :
            self.status_label.setText(str(status))
            
            # get the output and error msg
            self.output, self.err_msg = self.p.communicate()
            print('Output   :', self.output.decode("utf-8"))
            print('Error msg:', self.err_msg.decode("utf-8"))
            
            # emmit a signal when complete
            self.finished_signal.emit(False)


class Build_atlas_widget(QtGui.QWidget):
    def __init__(self, filename, config):
        super(Build_atlas_widget, self).__init__()
        
        self.filename    = filename
        self.config_dict = config
        self.initUI()

    def initUI(self):
        # get the output directory
        self.output_dir = os.path.split(self.filename)[0]
        self.config_filename = os.path.join(self.output_dir, 'build_atlas.ini')
        self.f = h5py.File(self.filename, 'r')
        self.f.close()

        # Make a grid layout
        ####################
        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        
        # make a vbox for the left column
        vst = QtGui.QVBoxLayout()
        
        # config widget
        ###############
        self.config_widget = Write_config_file_widget(self.config_dict, self.config_filename)
        vst.addWidget(self.config_widget)
        
        # get feature list
        ##################
        # dict of dict of lists
        self.features = self.read_features()
        
        # show frames with scatter plot
        ###############################
        self.init_display()
        
        # add feature grid button
        #########################
        #hst = QtGui.QHBoxLayout()
        #add_feature_grid_button   = QtGui.QPushButton('add feature grid')
        #add_feature_grid_button.clicked.connect( self.add_feature_grid )
        #self.add_feature_grid_lineedit = QtGui.QLineEdit()
        #self.add_feature_grid_lineedit.setText('4')
        #hst.addWidget(add_feature_grid_button)
        #hst.addWidget(self.add_feature_grid_lineedit)
        #vst.addLayout(hst)

        # add feature button
        ####################
        add_feature_button    = QtGui.QPushButton('add feature')
        add_feature_button.clicked.connect( self.add_feature )
        vst.addWidget(add_feature_button)

        # remove feature button
        #######################
        remove_feature_button = QtGui.QPushButton('remove feature')
        remove_feature_button.clicked.connect( self.remove_feature )
        vst.addWidget(remove_feature_button)
        
        # clear all button
        ##################
        clear_all_button = QtGui.QPushButton('clear all features')
        clear_all_button.clicked.connect( self.remove_all_features )
        vst.addWidget(clear_all_button)
        
        # write feature button
        ######################
        write_features_button = QtGui.QPushButton('write features')
        write_features_button.clicked.connect( self.write_features )
        vst.addWidget(write_features_button)

        # merge feature button
        ######################
        #merge_features_button = QtGui.QPushButton('merge features')
        #merge_features_button.clicked.connect( self.merge_features )
        #vst.addWidget(merge_features_button)
        
        # next duplicate button
        ##################
        next_pair_button = QtGui.QPushButton('next and duplicate')
        next_pair_button.clicked.connect( self.next_pair )
        vst.addWidget(next_pair_button)

        # next pair button
        ##################
        hst = QtGui.QHBoxLayout()
        nextp_button = QtGui.QPushButton('next pair')
        nextp_button.clicked.connect( self.nextp )
        prevp_button = QtGui.QPushButton('previous pair')
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
        self.run_button = QtGui.QPushButton('build atlas', self)
        self.run_button.clicked.connect(self.run_button_clicked)
        vst.addWidget(self.run_button)

        # run command button
        ####################
        self.run2_button = QtGui.QPushButton('add distortions', self)
        self.run2_button.clicked.connect(self.run2_button_clicked)
        vst.addWidget(self.run2_button)
        
        # add a spacer for the labels and such
        verticalSpacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
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
            config = self.config_dict['build_atlas']
            w = [config['window'], config['window']]
            
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
        self.config_widget.write_file()
    
        # Run the command 
        #################
        py = os.path.join('./', 'build_atlas.py')
        cmd = 'mpiexec -n 8 python ' + py + ' ' + self.filename + ' -c ' + self.config_filename
        self.run_command_widget.run_cmd(cmd)

    def run2_button_clicked(self):
        self.config_widget.write_file()
    
        # Run the command 
        #################
        py = os.path.join('./', 'add_distortions.py')
        cmd = 'python ' + py + ' ' + self.filename + ' -c ' + self.config_filename
        self.run_command_widget.run_cmd(cmd)

    def finished(self):
        self.remove_features_from_screen()
        self.features = self.read_features()
        self.update_display(self.frame1_index, self.frame2_index, update_atlas = True)

    def init_display(self):
        # get file info
        config = self.config_dict['build_atlas']
        roi    = config['roi']
        # show a frame
        self.f = h5py.File(self.filename, 'r')
        self.mask = self.f[config['mask']][roi[0]:roi[1], roi[2]:roi[3]]
        self.W    = self.f[config['whitefield']][roi[0]:roi[1], roi[2]:roi[3]]*self.mask
        self.W[self.W==0] = 1
        self.N      = self.f[config['frames']].shape[0]
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
        self.imgs_layout = QtGui.QGridLayout()

        vst1 = QtGui.QVBoxLayout()
        vst1.addWidget(self.frame1_imageView)
        vst1.addWidget(self.frame1_position_plotsW)

        vst2 = QtGui.QVBoxLayout()
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
        config = self.config_dict['build_atlas']
        roi    = config['roi']
        w = [config['window'], config['window']]
        # get the shift between frames
        f = h5py.File(self.filename)
        if (config['h5_group']+'/pix_positions') in f :
            pos = f[config['h5_group']+'/pix_positions'][()]
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
        config = self.config_dict['build_atlas']
        w = [config['window'], config['window']]
        
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
        config = self.config_dict['build_atlas']
        roi    = config['roi']
        w      = [config['window'], config['window']]
        
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
            if config['features'] in f :
                del f[config['features']]
            f[config['features']] = out
            f.close()
            print('\nfeatures written')
        else :
            print('\nno features to write')

    def read_features(self):
        config = self.config_dict['build_atlas']
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
        config = self.config_dict['build_atlas']
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

                
class Gui(QtGui.QTabWidget):
    def __init__(self):
        super(Gui, self).__init__()

    def initUI(self, filename):
        self.tabs = []

        self.setMovable(True)
        #self.setTabsClosable(True)

        # Show frames tab
        #################
        params = load_config(filename, name='build_atlas.ini')
        self.tabs.append( Build_atlas_widget(filename, params) )
        #self.tabs.append( Write_config_file_widget(params, 'MLL_260/build_atlas.ini'))
        self.addTab(self.tabs[-1], "build atlas")

def gui(filename):
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    app = QtGui.QApplication([])
    
    # Qt main window
    Mwin = QtGui.QMainWindow()
    Mwin.setWindowTitle(filename)
    
    cw = Gui()
    cw.initUI(filename)
    
    # add the central widget to the main window
    Mwin.setCentralWidget(cw)
    
    Mwin.show()
    app.exec_()

def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='calculate a basic stitch of the projection images')
    parser.add_argument('filename', type=str, \
                        help="file name of the *.pty file")
    
    args = parser.parse_args()
    
    # check that cxi file exists
    if not os.path.exists(args.filename):
        raise NameError('cxi file does not exist: ' + args.filename)
    
    return args

if __name__ == '__main__':
    args = parse_cmdline_args()
    
    gui(args.filename)
