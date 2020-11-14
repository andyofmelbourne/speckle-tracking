try :
    import PyQt5.QtWidgets as pyqt
except :
    import PyQt4.QtGui as pyqt

import h5py


class Show_h5_list_widget(pyqt.QWidget):
    def __init__(self, filename, names = None):
        super(Show_h5_list_widget, self).__init__()

        self.filename = filename
        self.names    = names
        
        # add the names to Qlist thing
        self.listWidget = pyqt.QListWidget(self)
        #self.listWidget.setMinimumWidth(self.listWidget.sizeHintForColumn(0))
        #self.listWidget.setMinimumHeight(500)
        
        # update list button
        ####################
        self.update_button = pyqt.QPushButton('update', self)
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
        self.layout = pyqt.QVBoxLayout()
        self.layout.addWidget(self.listWidget)
        self.layout.addWidget(self.update_button)
        
        # add the layout to the central widget
        self.setLayout(self.layout)

    def add_dataset_name(self, name, obj):
        print(name)
        names = self.names
        if isinstance(obj, h5py.Dataset):
            if ((names is None) or (names is not None and name in names)) \
                    and name not in self.dataset_names:
                self.dataset_names.append(name)
                self.dataset_items.append(pyqt.QListWidgetItem(self.listWidget))
                self.dataset_items[-1].setText(name)
    
    def update(self):
        self.remove_all()
        f = h5py.File(self.filename, 'r')
        f.visititems(self.add_dataset_name)
        f.close()

    def remove_all(self):
        item = 1
        while item is not None :
            item = self.listWidget.takeItem(0)
        item = None
        self.dataset_names = []
        self.dataset_items = []
