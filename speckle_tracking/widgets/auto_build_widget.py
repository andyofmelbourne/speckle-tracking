try :
    import PyQt5.QtWidgets as pyqt
except :
    import PyQt4.QtGui as pyqt

import sys, os
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
root = os.path.split(root)[0]

from config_editor_widget import Config_editor_Widget
from config_editor_widget import discover_config
from run_and_log_command  import Run_and_log_command
from run_and_log_command  import RunCommandWidget
from show_nd_data_widget  import Show_nd_data_widget

import multiprocessing
CPUS = min(multiprocessing.cpu_count() // 2, 8)

class Auto_build_widget(pyqt.QWidget):
    """
    ui layout is :
        | config editor |
        | run button    |
        ---run command widget---
    
    """
    def __init__(self, script_name, h5_fnam, config_fnam = '', config_dirs = ['process', 'bin'], mpi=False):
        super(Auto_build_widget, self).__init__()
        print('auto loading:', script_name)
        self.h5_fnam     = h5_fnam
        self.script_name = script_name
        self.mpi         = mpi
        
        self.config_fnams, self.config_output =  discover_config(script_name, \
                                                 h5_fnam, config_fnam, config_dirs)
        print(self.config_output)
        
        # Make the command 
        script    = self.script_name+'.py'
        self.cmd  = script + ' ' + self.h5_fnam + ' -c ' + self.config_output
        
        self.initUI()

    def initUI(self):
        # 
        vbox = pyqt.QVBoxLayout()

        vbox1 = pyqt.QVBoxLayout()
        # config widget
        ###############
        config_editor_widget = Config_editor_Widget(self.config_fnams, self.config_output)
        vbox1.addWidget(config_editor_widget)
        
        # run command widget
        ####################
        self.run_and_log_command = RunCommandWidget(cmd = self.cmd, watch='.log')
        self.run_and_log_command.display_signal.connect(lambda x : show_nd_data_widget.show(self.h5_fnam, x))
        
        # run command button
        ####################
        vbox1.addWidget(self.run_and_log_command.pushButton)

        hbox = pyqt.QHBoxLayout()
        hbox.addLayout(vbox1)
        
        # display widget
        ################
        show_nd_data_widget = Show_nd_data_widget()
        
        hbox.addWidget(show_nd_data_widget, stretch=1)
        vbox.addLayout(hbox)
        vbox.addWidget(self.run_and_log_command)
        
        self.setLayout(vbox)

    
