try :
    from PyQt5.QtWidgets import *
except :
    from PyQt4.QtGui import *



from show_h5_list_widget import Show_h5_list_widget
from show_nd_data_widget import Show_nd_data_widget

class View_h5_data_widget(QWidget):
    def __init__(self, filename, names = None):
        super(View_h5_data_widget, self).__init__()
        
        self.filename = filename
        self.names = names
            
        self.show_list_widget = Show_h5_list_widget(filename, names = names)
        self.plot1dWidget = Show_nd_data_widget()
        
        # send a signal when an item is clicked
        self.show_list_widget.listWidget.itemClicked.connect(self.dataset_clicked)

        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)

        # add the h5 datasets list
        layout.addWidget(self.show_list_widget)
        
        # add the 1d viewer 
        layout.addWidget(self.plot1dWidget, stretch=1)
        

    def dataset_clicked(self, item):
        name = str(item.text())
        
        # close the last image
        self.plot1dWidget.close()
        
        # load the new one
        self.plot1dWidget.show(self.filename, name)
        
    def update(self):
        self.show_list_widget.update()
        self.plot1dWidget.update()
