from PyQt4 import QtGui,QtCore
from Queue import Queue
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier as sgdc
from sklearn.neural_network import MLPClassifier as net
import numpy as np
import cv2
import os,sys
from time import sleep,time
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D,ZeroPadding2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from sknn.mlp import Classifier, Layer
import classifier_settings as clf_set
import psutil

def normalize_image(img):
    cv2.normalize(img,alpha=0,beta=255,norm_type=cv2.cv.CV_MINMAX)
    return img

def input_data(data_in,label,num_classes):
    QtGui.QApplication.processEvents()
    encod = LabelEncoder()
    labels = encod.fit_transform(label)
    labels = np_utils.to_categorical(labels,num_classes)
    data = np.array(data_in)/255.0
    return data,labels,encod

def get_decode(hsil,encod):
    val = hsil.argmax()
    predict = encod.inverse_transform(val)

    return predict

def files_count(dirc):
    num_files = 0
    if dirc:
        subdir = os.listdir(dirc)
        for sdir in subdir:
            files = os.listdir(dirc+'/'+sdir)
            num_files += len(files)

    return num_files,len(subdir)

class Window(QtGui.QDialog):
    def __init__(self):
        super(Window, self).__init__()
        self.initUI()
        self.init_var()
        
        mainLayout = QtGui.QGridLayout()
        mainLayout.addWidget(self.train_ui, 0, 0)
        self.mainLayout = mainLayout
        self.setLayout(self.mainLayout)
        self.setWindowTitle("Training Interface")
        self.setWindowIcon(QtGui.QIcon(os.getcwd()+'/Oxygen.ico'))

    def init_var(self):
        self.mem_total = round((psutil.virtual_memory()[0] * 1e-9),2)
        self.dirDBase = None
        self.model = None
        self.result = Queue()
        self.model_to_process = None
        self.classf = None
        self.imgDim = 150
        self.epoch = None
        
    def initUI(self):
        self.train_ui = QtGui.QGroupBox('Training Interface')
        
        self.close_btn = QtGui.QPushButton('Close')
        self.close_btn.clicked.connect(self.close)
        self.select_folder = QtGui.QPushButton('Select Folder')
        self.select_folder.clicked.connect(self.database)
        self.set_parameter = QtGui.QPushButton('Set Training Parameters')
        self.set_parameter.clicked.connect(self.classifier_settings)
        self.save_model = QtGui.QPushButton('Save Current Model')
        self.save_model.clicked.connect(self.model_save)
        self.memory_bar = QtGui.QProgressBar()
        self.memory_bar.setRange(0,100)
        self.cpu_bar = QtGui.QProgressBar()
        self.cpu_bar.setRange(0,100)
        self.found = QtGui.QLabel()
        self.status = QtGui.QLabel()
        self.stop_train = QtGui.QPushButton('Stop Training')
        self.stop_train.clicked.connect(self.stop_training)
        self.start_train = QtGui.QPushButton('Start Training')
        self.start_train.clicked.connect(self.start_training)
        self.start_train.setFixedWidth(100)
        self.blank_lbl = QtGui.QLabel('')
        self.result_lbl = QtGui.QLabel()
        self.result_lbl.setMaximumWidth(150)
        self.proc_time = QtGui.QLabel()
        
        self.comboBox = QtGui.QComboBox()
        self.comboBox.addItem('--Choose Classifier--',0)
        self.comboBox.addItem('Keras NN Model',1)
        self.comboBox.addItem('SGD Classifier',2)
        self.comboBox.addItem('MLP Perceptron NN',3)

        self.memory_lbl = QtGui.QLabel('Memory Usage')
        self.memory_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.cpu_lbl = QtGui.QLabel('CPU Usage')
        self.cpu_lbl.setAlignment(QtCore.Qt.AlignCenter)
        
        self.grid1 = QtGui.QGridLayout()
        self.grid1.addWidget(self.select_folder,0,1)
        self.grid1.addWidget(self.set_parameter,1,2)
        self.grid1.addWidget(self.save_model,0,0)
        self.grid1.addWidget(self.found,1,1)
        self.grid1.addWidget(self.start_train,1,0)
        self.grid1.addWidget(self.comboBox,1,1)
        self.grid1.addWidget(self.close_btn,0,2)
        self.grid1.addWidget(self.result_lbl,2,0)
        self.grid1.addWidget(self.proc_time,2,2)

        self.grid2 = QtGui.QGridLayout()
        self.grid2.addWidget(self.status,0,0)
        self.grid2.addWidget(self.memory_lbl,1,0)
        self.grid2.addWidget(self.cpu_lbl,1,1)
        self.grid2.addWidget(self.memory_bar,2,0)
        self.grid2.addWidget(self.cpu_bar,2,1)
        self.grid2.addWidget(self.stop_train,0,1)
        

        self.main_widget = QtGui.QWidget()
        self.main_widget.setLayout(self.grid1)
        self.proc_widget = QtGui.QWidget()
        self.proc_widget.setLayout(self.grid2)
        self.proc_widget.setVisible(False)
        

        grid = QtGui.QGridLayout()
        grid.addWidget(self.main_widget)
        grid.addWidget(self.proc_widget)
        
        self.train_ui.setLayout(grid)

    def database(self):
        dirName = QtGui.QFileDialog.getExistingDirectory(self, "Choose Directory",
                QtCore.QDir.currentPath())
        if not(dirName):
            return
        self.dirDBase = str(dirName)
        num_files,num_classes = files_count(self.dirDBase)
        self.num_files = num_files
        self.num_classes = num_classes
        db = self.dirDBase.split("/")
        db = '.../'+db[-2]+'/'+db[-1]+'\n'
        teks = [str(num_classes)+' classes \n',str(num_files)+' images']
        self.result_lbl.setText(db+teks[0]+teks[1])
        del teks,db
        self.state(True)

    def collect_data(self):
        if not(self.dirDBase):
            return
        imgDim = self.imgDim
        data=[]
        label=[]
        subdir = os.listdir(self.dirDBase)
        self.num_classes = len(subdir)
        c = 0
        self.status.setVisible(True)
        self.status.setText('Collecting and Processing image...')
        for sd in subdir:
            imgPath = os.listdir(self.dirDBase+'/'+sd)
            for jpg in imgPath:
                os.chdir(self.dirDBase+'/'+sd)
                image = cv2.imread(jpg)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = normalize_image(gray)
                image = cv2.resize(image,(imgDim,imgDim))
                image = np.expand_dims(image, axis=0)
                QtGui.QApplication.processEvents()
                c += 1
                self.status.setText('Data Processed : {}/{}'.format(str(c),
                                                                    str(self.num_files)))
                data.append(image)
                label.append(sd)
        self.status.setText('Preparing Model...')

        return data,label

    def classifier_settings(self):
        if self.dirDBase is None:
            self.result_lbl.setText('Please input your data first')
            return

        clf = self.comboBox.currentIndex()
        self.classf = clf-1
        arg = [clf-1,self.num_classes]
        if clf==1:
            win = clf_set.Window(arg[0],arg[1])
            win.exec_()
            if win.result()==0:
                data = win.save()
            del win
            if data is None:
                self.result_lbl.setText('No model found')
                return
            else:
                self.result_lbl.setText('Keras model build obtained.')
                self.model_to_process = data
            
        elif clf==2:
            win = clf_set.Window(arg[0],arg[1])
            win.exec_()
            if win.result()==0:
                data = win.save()
            del win
            if data is None:
                self.result_lbl.setText('No model found')
                return
            else:
                self.result_lbl.setText('SGDClassifier model build\nobtained.')
                self.model_to_process = data
                
        elif clf==3:
            win = clf_set.Window(arg[0],arg[1])
            win.exec_()
            if win.result()==0:
                data = win.save()
            del win
            if data is None:
                self.result_lbl.setText('No model found')
                return
            else:
                self.result_lbl.setText('MLP Perceptron model build \nobtained.')
                self.model_to_process = data
                
        else:
            self.result_lbl.setText('Choose training classifier')
            return

    def state(self,v):
        self.main_widget.setVisible(v)
        self.proc_widget.setVisible(not(v))

            
    def confirm_validate(self):
        msgBox = QtGui.QMessageBox()
        msgBox.setWindowTitle('Validating setting')
        msgBox.setText('Do You need to validate the model? ')
        Ok = msgBox.addButton(QtGui.QMessageBox.Ok)
        Cancel = msgBox.addButton(QtGui.QMessageBox.Cancel)
        No = msgBox.addButton('No',QtGui.QMessageBox.ActionRole)
        msgBox.exec_()

        if msgBox.clickedButton()==Ok:
            return True
        elif msgBox.clickedButton()==Cancel:
            self.result_lbl.setText('Training canceled')
            return None
        else:
            return False

    def start_training(self):
        self.on_train=True
        if not(self.dirDBase):
            return
        if self.model_to_process is None:
            return
        self.state(False)
        self.status.setVisible(True)
        
        self.status.setText('Prepare training....')
        data_in,label_in = self.collect_data()
        self.status.setText('Training in progress...(Please wait)')
        argd = [data_in,label_in,os.listdir(self.dirDBase),self.result]
        if self.on_train==False:
            return
        val = self.confirm_validate()
        if val==None:
            return
        arga = [val,self.classf]
        self.epoch = self.model_to_process[-1]
            
        self.threads = Worker(argd,self.model_to_process,arga)
        self.monitor = cpuUsage()
        self.training_thread()
        del data_in,label_in
            
    def get_model(self):
        k = self.result.get()
        self.model = k[0]
        score = k[1]
        g = k[2]
        self.classf = k[2]
        if score is not None:            
            if g==0:
                loss = str(round(score[0],4))
                acc = str(100*round(score[1],5))
                self.encod = k[3]
                self.result_lbl.setText('Training Done. \n(Accuracy : {} and \nloss : {})'.format(acc,loss))
            else:
                self.result_lbl.setText('Training Done. \n(Accuracy : {:.4f})'.format(100*score))

        else:
            if g==0:
                self.encod = k[3]
            self.result_lbl.setText('Training Done.')

        self.threads.break_loop = True
        self.monitor.henti = True
        self.monitor.quit()
        self.threads.quit()
        self.state(True)
        self.memory_bar.setValue(0)
        self.cpu_bar.setValue(0)
        self.status.setText('')

    def model_save(self):
        if (self.dirDBase)and(self.model):
            dirPath = os.path.split(self.dirDBase)[0]
            if self.classf==0 : newdir='Keras_'
            elif self.classf==1 : newdir='SGDC_'
            elif self.classf==2 : newdir='MLP_'
            namafile,ok = QtGui.QInputDialog.getText(self,'Nama File','Nama File',
                                                  QtGui.QLineEdit.Normal)
            if ok:
                os.chdir(dirPath)
                os.mkdir(newdir+str(namafile)+'_build')
                os.chdir(newdir+str(namafile)+'_build')
                if self.classf==0:
                    QtGui.QApplication.processEvents()
                    self.model.save_weights(str(namafile)+'_weigths.h5')
                    QtGui.QApplication.processEvents()
                    self.model.save(str(namafile)+'_full.h5')
                    pickle.dump(self.model_to_process,
                                open(str(namafile)+'_settings.pickle', 'wb'))
                    pickle.dump([self.num_classes,self.encod],
                                open(str(namafile)+'_params.pickle', 'wb'))
                else:
                    pickle.dump(self.model, open(str(namafile)+'.sav', 'wb'))
                    pickle.dump(self.model_to_process,
                                open(str(namafile)+'_settings.pickle', 'wb'))
                self.result_lbl.setText('Model telah disimpan')
        else:
            self.result_lbl.setText('No model to be saved')
            return
                                    
        
    def output_stream(self,val):
        self.status.setVisible(True)
        if val==0:
            self.status.setText('Training Done, validating model...(please wait)')
        else:
            self.status.setText('Training...(Epoch : {}/{})'.format(str(val),str(self.epoch)))

    def mem_monitor(self,v):
        self.memory_bar.setValue(v)
        use = round(psutil.virtual_memory()[3]*1e-9,2)
        self.memory_lbl.setText('Memory Usage ('+str(use)+'GB/'
                                +str(self.mem_total)+'GB)')
    
    def cpu_monitor(self,v):
        self.cpu_bar.setValue(v)
        
    def training_thread(self):
        if not self.threads.isRunning():
            self.threads.exiting = False
            self.monitor.henti = False
            self.threads.output.connect(self.output_stream)
            self.monitor.output1.connect(self.cpu_monitor)
            self.monitor.output2.connect(self.mem_monitor)
            self.threads.finished.connect(self.get_model)
            self.monitor.start()
            self.threads.start()

    def stop_training(self):
        self.on_train=False
        msgBox = QtGui.QMessageBox()
        msgBox.setWindowTitle('Stop Current Training Confirmation')
        msgBox.setText("Stop Current Training Progress")
        msgBox.setInformativeText("Apakah anda yakin ingin menghentikan training?")
        Ok = msgBox.addButton(QtGui.QMessageBox.Ok)
        Cancel = msgBox.addButton(QtGui.QMessageBox.Cancel)
        msgBox.exec_()

        if msgBox.clickedButton()==Ok:
            if self.classf == 0:
                self.threads.quit()
                QtGui.QMessageBox.information(self, "Face Recognition Helper",
                                "Keras training method cannot be terminated "
                                "please wait until the training done.")
                self.status.setText('Training in progress...(Please wait)')
                return
            else:
                self.threads.break_loop = True
                self.threads.quit()
        elif msgBox.clickedButton()==Cancel:
            return

        self.state(True)
        
            

class Worker(QtCore.QThread):
    output = QtCore.pyqtSignal(int)

    def __init__(self,argd,argm,arga):
        QtCore.QThread.__init__(self)
        self.num_classes = len(argd[2])
        self.data = argd[0]
        self.label = argd[1]
        self.result = argd[3]
        self.epoch = argm[len(argm)-1]
        self.train_data = argd[2]
        self.clfState = arga[1]
        self.break_loop=False
        self.val = arga[0]
        self.argm = argm
        
##        argm berisi model parameters
##        argd berisi database parameters
##        arga berisi setting tmbahan(validasi logic,jenis classifier)
        
    def run(self):
        while self.exiting==False:
            g = self.clfState
            if g==0:
                self.Keras_Model()
            elif g==1:
                self.SGD_Model()
            elif g==2:
                self.SGD_Model('mlp')
            self.exiting = True

    def Keras_Model(self):
        data_in,label_in,encoder = input_data(self.data,self.label,
                                              self.num_classes)
        
        model = Sequential()
        for layer in self.argm[0]:
            model.add(layer)
            
        QtGui.QApplication.processEvents()
        iterasi = self.argm[2]
        opt = self.argm[1]
        model.compile(loss='binary_crossentropy',
                      optimizer=opt, metrics=['accuracy'])
        QtGui.QApplication.processEvents()
        model.fit(data_in,label_in,nb_epoch=iterasi,batch_size=self.argm[3],verbose=2)

        if self.val:
            self.output.emit(0)
            (loss,accuracy) = model.evaluate(data_in, label_in,batch_size=64, verbose=0)
            score = [loss,accuracy]
            self.result.put([model,score,self.clfState,encoder])
        else:
            self.result.put([model,None,self.clfState,encoder])
            
        del data_in,self.data
        del label_in,self.label
        del model

    def SGD_Model(self,choices='sgd'):
        if choices=='sgd':
            model = self.argm[0]
        else:
            model = self.argm[0]
        
        for i in range(self.argm[1]):
            if self.break_loop==True:
                continue
            model.partial_fit(self.data,self.label,classes = self.train_data)
            self.output.emit(i+1)

        if self.val:
            self.output.emit(0)
            score = model.score(self.data,self.label)
            self.result.put([model,score,self.clfState])
        else:
            self.result.put([model,None,self.clfState])
        del self.data
        del self.label
        del model


class cpuUsage(QtCore.QThread):
    output1 = QtCore.pyqtSignal(float)
    output2 = QtCore.pyqtSignal(float)

    def __init__(self):
        QtCore.QThread.__init__(self)

        
    def run(self):
        while self.henti==False:
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            self.output1.emit(cpu)
            self.output2.emit(mem[2])
            
        
if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)
    dialog = Window()
    dialog.show()

    sys.exit(app.exec_())
