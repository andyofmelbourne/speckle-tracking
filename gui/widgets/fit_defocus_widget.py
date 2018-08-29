try :
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
except :
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

import pyqtgraph as pg

import sys, os
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
root = os.path.split(root)[0]

from config_editor_widget import Config_editor_Widget
from config_editor_widget import discover_config
from run_and_log_command  import Run_and_log_command
from show_nd_data_widget  import Show_nd_data_widget

import multiprocessing
CPUS = min(multiprocessing.cpu_count() // 2, 8)


class Thon_widget(QWidget):
    def __init__(self):
        super(Thon_widget, self).__init__()
        self.initUI()
    
    def initUI(self):
        # 
        vbox = QVBoxLayout()
        splitter = QSplitter(Qt.Vertical)

        self.thonV = pg.ImageView(view = pg.PlotItem(title = 'Thon rings + fit'))
        self.thonV.ui.menuBtn.hide()
        self.thonV.ui.roiBtn.hide()
        splitter.addWidget(self.thonV)
        
        plot = self.frame1_position_plotsW = pg.PlotWidget()

        title = 'Thon 1d (red), fit (green) and quality of fit (white)'
        self.thonP = pg.PlotWidget(bottom='pixel radius', left='intensity', title = title)
        splitter.addWidget(self.thonP)
        
        title = 'Fit error vs defocus'
        self.thon_errP = pg.PlotWidget(bottom='focus->sample (m)', left='error', title = title)
        splitter.addWidget(self.thon_errP)

        vbox.addWidget(splitter)
        self.setLayout(vbox)
    
    def update_display(self, thon=None, fits=None, errs=None, defocus=None):
        """
        thon is a 2d array showing Thon rings
        fits is a 2d array (3, N) showing 
            [0, :] flattened thon ring
            [1, :] fit Thon ring
            [2, :] quality of fit
        errs is a 2d array (2, N) with
            [0, :] error(defocus)
            [1, :] defocus values
        """
        if thon is not None :
            self.thonV.setImage(thon, autoRange = False, autoLevels = False, autoHistogramRange = False)

        if fits is not None :
            pens = [(255, 150, 150), (150, 255, 150), (150, 150, 150)]
            self.thonP.clear()
            for fit, pen in zip(fits, pens):
                self.thonP.plot(fit, pen=pen)
        
        if errs is not None :
            self.thon_errP.clear()
            self.thon_errP.setTitle('Fit error vs defocus: ' + str(defocus)) 
            self.thon_errP.plot(errs[1], errs[0])

        



class Fit_defocus_widget(QWidget):
    """
    ui layout is :
        | config editor |
        | run button    |
        ---run command widget---
    
    """
    def __init__(self, script_name, h5_fnam, config_fnam = '', config_dirs = ['/process/',], mpi=False):
        super(Fit_defocus_widget, self).__init__()
        print('auto loading:', script_name)
        self.h5_fnam     = h5_fnam
        self.script_name = script_name
        self.mpi         = mpi
        
        self.config_fnams, self.config_output =  discover_config(script_name, \
                                                 h5_fnam, config_fnam, config_dirs)
        
        self.initUI()

    def initUI(self):
        # 
        vbox = QVBoxLayout()

        vbox1 = QVBoxLayout()
        # config widget
        ###############
        config_editor_widget = Config_editor_Widget(self.config_fnams, self.config_output)
        vbox1.addWidget(config_editor_widget)
        
        # run command button
        ####################
        run_button = QPushButton('Run', self)
        run_button.clicked.connect(self.run_button_clicked)
        vbox1.addWidget(run_button)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox1)
        
        # display widget
        ################
        show_nd_data_widget = Show_nd_data_widget()
        
        self.thonW = Thon_widget()
        hbox.addWidget(self.thonW, stretch=1)
        vbox.addLayout(hbox)
        
        # run command widget
        ####################
        self.run_and_log_command = Run_and_log_command()
        self.run_and_log_command.finished_signal.connect(self.finished)
        self.run_and_log_command.display_signal.connect(lambda x : show_nd_data_widget.show(self.h5_fnam, x))
        vbox.addWidget(self.run_and_log_command)
        
        self.setLayout(vbox)

        # update display
        self.update_display()
    
    def update_display(self):
        import h5py
        sys.path.insert(0, os.path.join(root, 'utils'))
        import config_reader
        # re-read the output config file 
        # in case the output group has changed (config_output)
        params, fnam = config_reader.config_read([self.config_output] + self.config_fnams)
        thon, fits, errs, defocus = None, None, None, None
        with h5py.File(self.h5_fnam, 'r') as f:
            if params[self.script_name]['h5_group'] in f :
                g = f[params[self.script_name]['h5_group']]
                if 'thon_with_fit' in g :
                    thon = g['thon_with_fit'][()].T
                if 'fits' in g :
                    fits = g['fits'][()]
                if 'errs' in g :
                    errs = g['errs'][()]
                if 'defocus' in g :
                    defocus = g['defocus'][()]

        self.thonW.update_display(thon, fits, errs, defocus)
    
    def finished(self):
        print('done!')
        self.update_display()
    
    def run_button_clicked(self):
        # Run the command 
        #################
        script  = os.path.join(root, 'process/'+self.script_name+'.py')
        script  = os.path.relpath(script, os.getcwd())
        if self.mpi is True :
            cmd = 'mpiexec -np '+str(CPUS)+' python -W ignore ' + script + ' ' + self.h5_fnam + ' -c ' + self.config_output
        else :
            cmd = 'python ' + script + ' ' + self.h5_fnam + ' -c ' + self.config_output
        self.run_and_log_command.run_cmd(cmd)
