try :
    from PyQt5.QtWidgets import *
except :
    from PyQt4.QtGui import *

import sys, os
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
root = os.path.split(root)[0]

from config_editor_widget import Config_editor_Widget
from run_and_log_command  import Run_and_log_command
from show_nd_data_widget  import Show_nd_data_widget

class Auto_build_widget(QWidget):
    """
    ui layout is :
        | config editor |
        | run button    |
        ---run command widget---
    
    """
    def __init__(self, script_name, h5_fnam, config_fnam = '', config_dirs = ['/process/',]):
        super(Auto_build_widget, self).__init__()
        self.h5_fnam     = h5_fnam
        self.script_name = script_name
        
        # config output file is [h5_data directory]/[sript_name].ini
        #
        self.config_output = os.path.split(h5_fnam)[0] + '/' + script_name + '.ini'
        
        # config search order is config_fnam, h5_fnam dir, config_dirs..
        # 
        self.config_fnams = [config_fnam,]
        self.config_fnams.append(self.config_output)
        for cd in config_dirs :
            self.config_fnams.append(root + cd + '/' + script_name + '.ini') 
        
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
        
        hbox.addWidget(show_nd_data_widget, stretch=1)
        vbox.addLayout(hbox)
        
        # run command widget
        ####################
        self.run_and_log_command = Run_and_log_command()
        self.run_and_log_command.finished_signal.connect(self.finished)
        self.run_and_log_command.display_signal.connect(lambda x : show_nd_data_widget.show(self.h5_fnam, x))
        vbox.addWidget(self.run_and_log_command)
        
        self.setLayout(vbox)

    def finished(self):
        print('done!')
    
    def run_button_clicked(self):
        # Run the command 
        #################
        script  = os.path.join(root, 'process/'+self.script_name+'.py')
        script  = os.path.relpath(script, os.getcwd())
        cmd = 'python ' + script + ' ' + self.h5_fnam + ' -c ' + self.config_output
        self.run_and_log_command.run_cmd(cmd)
