try :
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import pyqtSignal, QTimer
except :
    from PyQt4.QtGui import *
    from PyQt4.QtCore import pyqtSignal, QTimer

import sys
from threading  import Thread
from queue import Queue, Empty


def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

class Run_and_log_command(QWidget):
    """
    run a command and send a signal when it complete, or it has failed.

    use a Qt timer to check the process
    
    realtime streaming of the terminal output has so proved to be fruitless
    """
    finished_signal = pyqtSignal(bool)
    
    def __init__(self):
        super(Run_and_log_command, self).__init__()
        
        self.polling_interval = 10.
        self.initUI()
        
    def initUI(self):
        """
        Just setup a qlabel showing the shell command
        and another showing the status of the process
        """
        # Make a grid layout
        #layout = QGridLayout()
        hbox = QHBoxLayout()
        
        # add the layout to the central widget
        self.setLayout(hbox)
        
        # show the command being executed
        self.command_label0 = QLabel(self)
        self.command_label0.setText('<b>Command:</b>')
        self.command_label  = QLabel(self)
        #self.command_label.setMaximumSize(50, 250)
         
        # show the status of the command
        self.status_label0  = QLabel(self)
        self.status_label0.setText('<b>Status:</b>')
        self.status_label   = QLabel(self)
        
        # add to layout
        hbox.addWidget(self.status_label0)
        hbox.addWidget(self.status_label)
        hbox.addWidget(self.command_label0)
        hbox.addWidget(self.command_label)
        hbox.addStretch(1)

    def run_cmd(self, cmd):
        from subprocess import PIPE, Popen
        import shlex
        self.command_label.setText(cmd)
        self.status_label.setText('running the command')
        self.p = Popen(shlex.split(cmd), stdout = PIPE, stderr = PIPE)
        self.q = Queue()
        t = Thread(target=enqueue_output, args=(self.p.stdout, self.q))
        # make sure the thread dies with the program
        t.daemon = True
        t.start()
        
        # start a Qt timer to update the status
        QTimer.singleShot(self.polling_interval, self.update_status)
    
    def update_status(self):
        status = self.p.poll()
        if status is None :
            self.status_label.setText('Running')
            
            # non blocking readline
            try :
                line = self.q.get_nowait()
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
            except Empty :
                pass
            
            # start a Qt timer to update the status
            QTimer.singleShot(self.polling_interval, self.update_status)
        
        elif status is 0 :
            self.status_label.setText('Finished')
            
            # emmit a signal when complete
            self.finished_signal.emit(True)
            
        else :
            self.status_label.setText(str(status))
            
            # get the output and error msg
            for line in iter(self.p.stderr.readline, b''):
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
            
            # emmit a signal when complete
            self.finished_signal.emit(False)
