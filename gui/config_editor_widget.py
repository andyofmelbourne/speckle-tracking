import signal

try :
    from PyQt5.QtWidgets import *
except :
    from PyQt4.QtGui import *

import config_reader

class QForm_w(QWidget):
    """
    """
    def __init__(self, update_config):
        super(QForm_w, self).__init__()
        
        self.layout = QFormLayout()
        self.setLayout(self.layout)
        self.is_vis = True
        self.l_eds  = []
        self.keys   = []
        self.update_config = update_config
    
    def addRow(self, key, val, doc=None):
        label = QLabel(key)
        if doc is not None :
            label.setToolTip(doc)
        self.l_eds.append(QLineEdit(str(val)))
        self.l_eds[-1].editingFinished.connect(self.update_config)
        self.keys.append(key)
        self.layout.addRow(label, self.l_eds[-1])
    
    def toggle_disp(self):
        if self.is_vis :
            self.hide()
        else :
            self.show()
        self.is_vis = not self.is_vis

    def get_vals(self):
        vals = []
        for key, le in zip(self.keys, self.l_eds):
            vals.append( (key, str(le.text()).strip()))
        return vals

class Config_editor_Widget(QWidget):
    """
    """
    def __init__(self, config_fnam, output_filename = None):
        super(Config_editor_Widget, self).__init__()
        
        # read the input config file
        self.config, fnam = config_reader.config_read(config_fnam, True)
        
        # set the output filename 
        if output_filename is None :
            self.output_filename = fnam
        else :
            self.output_filename = output_filename
        
        self.initUI()
    
    def initUI(self):
        # Make a vertical stack
        layout = QVBoxLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # add the output config filename 
        ################################    
        fnam_label = QLabel(self)
        fnam_label.setText('<b>'+self.output_filename+'</b>')
        layout.addWidget(fnam_label)
        
        self.forms  = []
        self.groups = []
        for group in self.config.keys():
            adv_options = []
            # add the params form widget
            ############################
            self.forms.append(QForm_w(self.update_config))
            for key in self.config[group].keys():
                val, doc, adv = self.config[group][key]
                if adv is False :
                    self.forms[-1].addRow(key, val, doc)
                else :
                    adv_options.append((key, val, doc))
            
            # add the group collapsible widget
            ##################################  
            self.groups.append(group)
            group_label = QPushButton(group, self)
            group_label.clicked.connect(self.forms[-1].toggle_disp)
            
            layout.addWidget(group_label)
            layout.addWidget(self.forms[-1])
            
            # repeat for advanced options 
            #############################
            if len(adv_options) > 0 :
                self.forms.append(QForm_w(self.update_config))
                for key, val, doc in adv_options :
                    self.forms[-1].addRow(key, val, doc)
                
                self.forms[-1].toggle_disp()
                
                # add the group collapsible widget
                ##################################  
                self.groups.append(group)
                group_label = QPushButton(group+'-advanced', self)
                group_label.clicked.connect(self.forms[-1].toggle_disp)
                
                layout.addWidget(group_label)
                layout.addWidget(self.forms[-1])
         
        verticalSpacer = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(verticalSpacer)

    def update_config(self):
        for group, form in zip(self.groups, self.forms):
            for key, val in form.get_vals():
                oldval, doc, adv = self.config[group][key]
                self.config[group][key] = (val, doc, adv)
        
        config_reader.config_write(self.config, self.output_filename, True)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    app = QApplication([])
    
    # Qt main window
    Mwin = QMainWindow()
    Mwin.setWindowTitle('config editor')
    
    config_widget = Config_editor_Widget('example.ini', 'example_output.ini')
    
    # add the central widget to the main window
    Mwin.setCentralWidget(config_widget)
    
    print('app exec')
    Mwin.show()
    app.exec_()
