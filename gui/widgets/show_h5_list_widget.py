try :
    from PyQt5.QtWidgets import *
except :
    from PyQt4.QtGui import *

import h5py


class Show_h5_list_widget(QWidget):
    def __init__(self, filename, names = None):
        super(Show_h5_list_widget, self).__init__()

        self.filename = filename
        self.names    = names
        
        # add the names to Qlist thing
        self.listWidget = QListWidget(self)
        #self.listWidget.setMinimumWidth(self.listWidget.sizeHintForColumn(0))
        #self.listWidget.setMinimumHeight(500)
        
        # update list button
        ####################
        self.update_button = QPushButton('update', self)
        self.update_button.clicked.connect(self.update)

        # get the list of groups and items
        self.dataset_names = [] 
        self.dataset_items = [] 
        
        f = h5py.File(filename, 'r')
        f.visititems(self.add_dataset_name)
        f.close()

        self.initUI()
    
    def initUI(self):
        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.listWidget)
        layout.addWidget(self.update_button)
        
        # add the layout to the central widget
        self.setLayout(layout)

    def add_dataset_name(self, name, obj):
        names = self.names
        if isinstance(obj, h5py.Dataset):
            if ((names is None) or (names is not None and name in names)) \
                    and name not in self.dataset_names:
                self.dataset_names.append(name)
                self.dataset_items.append(QListWidgetItem(self.listWidget))
                self.dataset_items[-1].setText(name)
    
    def update(self):
        f = h5py.File(self.filename, 'r')
        f.visititems(self.add_dataset_name)
        f.close()
