try :
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import pyqtSignal, QTimer, QProcess, QFileSystemWatcher
except :
    from PyQt4.QtGui import *
    from PyQt4.QtCore import pyqtSignal, QTimer, QProcess, QFileSystemWatcher

import sys, os, time
from threading  import Thread
from queue import Queue, Empty

def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()


class WatchFileWidget(QWidget):
    display_signal = pyqtSignal(str)
    
    def __init__(self, fnam = '.log'):
        super(WatchFileWidget, self).__init__()
        
        self.fnam = fnam
        self.last_read = None
        self.min_time = 1e-1
        
    def start(self):
        # if the file does not exist
        # create it then watch it
        if not os.path.exists(self.fnam):
            with open(self.fnam, 'w') as f:
                pass
        
        # watch file
        self.w = QFileSystemWatcher([self.fnam])
        self.w.fileChanged.connect(self.fileChanged)

    def fileChanged(self):
        # if the file was read less than self.min_time ago, then skip
        if (self.last_read is not None) and (time.time()-self.last_read)<self.min_time :
            return
        
        # read the file and print output
        with open(self.fnam) as f:
            lines = f.readlines()
            last_line = lines[-1]
            print('last_line:', last_line)
            self.last_read = time.time()
            self.display_signal.emit(last_line.split('display:')[1].strip())
    
    def stop(self):
        self.w.removePath(self.fnam)
    
class RunCommandWidget(QWidget):
    
    finished_signal = pyqtSignal(bool)
    display_signal = pyqtSignal(str)
    
    def __init__(self, cmd = 'python test_run.py', watch=None):
        super(RunCommandWidget, self).__init__()

        if watch is not None and watch is not False :
            self.fnam = watch
            self.watch = WatchFileWidget(fnam=self.fnam)
            self.display_signal = self.watch.display_signal
        else :
            self.fnam = None
        
        self.cmd = cmd
        self.polling_interval = 10.
        self.initUI()
    
    def initUI(self):
        # Run button
        self.pushButton = QPushButton('Run')
        self.pushButton.clicked.connect(self.run_cmd)
         
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
        
    def run_cmd(self):
        # update button
        self.pushButton.setEnabled(0)
        self.pushButton.setText("Processing")
        
        command = self.cmd.split(" ")[0]
        args    = self.cmd.split(" ")[1:]
        self.process = QProcess(self)
        self.process.setProcessChannelMode(2)
        self.process.finished.connect(self.onFinished)
        #process.startDetached(command, args)
        
        # update text
        self.command_label.setText(self.cmd)
        self.status_label.setText('Processing')
        
        # watch file
        if self.fnam is not None :
            self.watch.start()
        
        # run cmd
        self.process.start(command, args)
    
    def onFinished(self, exitCode, exitStatus):
        if self.fnam is not None :
            self.watch.stop()
        
        if int(exitStatus) == 0 :
            self.status_label.setText('Finished')
        else :
            self.status_label.setText('Error')
        
        self.pushButton.setText("Run")
        self.pushButton.setEnabled(True)
        self.finished_signal.emit(True)


class Run_and_log_command(QWidget):
    """
    run a command and send a signal when it complete, or it has failed.

    use a Qt timer to check the process
    
    realtime streaming of the terminal output has so proved to be fruitless
    """
    finished_signal = pyqtSignal(bool)
    display_signal = pyqtSignal(str)
    
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
                
                # emmit a signal on 'display:'
                line = line.decode("utf-8")
                if 'display:' in line:
                    self.display_signal.emit(line.split('display:')[1].strip())
            except Empty :
                pass
            
            # start a Qt timer to update the status
            QTimer.singleShot(self.polling_interval, self.update_status)
        
        elif status is 0 :
            self.status_label.setText('Finished')
            
            # non blocking readline
            try :
                line = self.q.get_nowait()
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
                
                # emmit a signal on 'display:'
                line = line.decode("utf-8")
                if 'display:' in line:
                    self.display_signal.emit(line.split('display:')[1].strip())
            except Empty :
                pass
            
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
