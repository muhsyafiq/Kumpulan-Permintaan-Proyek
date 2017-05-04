from PyQt4 import QtGui,QtCore
import numpy as np
import cv2
import dlib
import os,sys
from time import sleep,time
import pickle
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier as sgdc
from sklearn.neural_network import MLPClassifier as net
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential,load_model
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


FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))
ALL_POINTS = list(range(0,68))

INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
OUTER_EYES_AND_NOSE = [36, 45, 33]

face_landmark_tensor_path = os.getcwd()+'/face_tensor.txt'


# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

predictor_path = os.getcwd()+'/shape_predictor_68_face_landmarks.dat'

class_build = {'Keras':0,'SGDC':1,'MLP':2}

def keras_model_build(argm,wpath):
    model = Sequential()
    for layer in argm[0]:
        model.add(layer)

    model.load_weights(wpath)

    iterasi = argm[2]
    opt = argm[1]
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

def align_face(Img,imgDim,landpoints, landIndices):
    landmarks = np.float32(landpoints)
    landInc = np.array(landIndices)
    face_tensor = np.float32(np.loadtxt(face_landmark_tensor_path))
    M = cv2.getAffineTransform(landmarks[landInc],imgDim * face_tensor[landInc])
    output = cv2.warpAffine(Img,M, (imgDim,imgDim))
    
    return output

def getFaceLoc(img):
    detector = dlib.get_frontal_face_detector()
    dets = detector(img,1)
    return dets

predictor = dlib.shape_predictor(predictor_path)
def get_Face(img,dets,imgDim,indices=ALL_POINTS):
    
    imageData = {}
    for i in range(len(dets)):
        shape = predictor(img, dets[i])
        land = list(map(lambda p: (p.x, p.y), shape.parts()))
        face = align_face(img,imgDim,land,indices)
        imageData[i] = face

    return imageData

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

class Window(QtGui.QDialog):
    def __init__(self,m):
        super(Window, self).__init__()
        
        self.mode = m
        self.initUI()
        self.setWindowIcon(QtGui.QIcon(os.getcwd()+'/Oxygen.ico'))
        
        
        
    def initUI(self):
        self.Indices = INNER_EYES_AND_BOTTOM_LIP
        self.dirDBase = None
        self.model = None
        self.model_to_process = None
        self.imgDim = 150
        self.epoch = None
        self.data = {}
        self.label_predicted= None
        self.rect = {}
        self.count = 0
        self.save_dirName = self.namafile = None
        self.delayTime = 1
        self.cur_delay = 1.0/20
        if self.mode == 0:
            self.pict_mode()
        else:
            self.playing=True
            self.classf = None
            self.current_device = 0
            self.real_time_predict = False
            self.webcam_mode()

    def pict_mode(self):
        self.view_panel = QtGui.QLabel('View Panel')
        self.view_panel.setAlignment(QtCore.Qt.AlignCenter)
        self.view_panel.setFixedSize(500,500)
        close_btn = QtGui.QPushButton('Close')
        open_file = QtGui.QPushButton('Open File')
        open_file.clicked.connect(self.buka_file)
        predict_btn = QtGui.QPushButton('Predict')
        load_model = QtGui.QPushButton('Load Model')
        save_crop = QtGui.QPushButton('Save Cropped Image')
        save_crop.clicked.connect(self.save_cropped)
        self.process_lbl = QtGui.QLabel('')

        close_btn.clicked.connect(self.out)
        predict_btn.clicked.connect(self.predict)
        load_model.clicked.connect(self.model_load)

        grid1 = QtGui.QHBoxLayout()
        grid1.addWidget(save_crop)
        grid1.addWidget(predict_btn)
        grid1.addWidget(open_file)
        grid1.addWidget(load_model)
        grid1.addWidget(close_btn)
        
        self.pict_widget = QtGui.QWidget()
        self.pict_widget.setLayout(grid1)

        grid = QtGui.QGridLayout()
        grid.addWidget(self.view_panel,0,0,1,2)
        grid.addWidget(self.pict_widget,1,1)
        grid.addWidget(self.process_lbl,2,0)
        
        self.setLayout(grid)
        self.show()
       
        self.setWindowTitle("Predict Image Interface")

    def webcam_mode(self):
        self.view_panel = QtGui.QLabel('View Panel')
        self.view_panel.setAlignment(QtCore.Qt.AlignCenter)
        self.view_panel.setFixedSize(500,500)
        close_btn = QtGui.QPushButton('Close')
        self.predict_btn = QtGui.QPushButton('Predict')
        webcam_dev = QtGui.QPushButton('Change Device')
        webcam_dev.clicked.connect(self.device_change)
        load_model = QtGui.QPushButton('Load Model')
        load_model.clicked.connect(self.model_load)
        self.stop_btn = QtGui.QPushButton('Stop Predict')
        self.process_lbl = QtGui.QLabel('')

        close_btn.clicked.connect(self.out)
        self.predict_btn.clicked.connect(self.predict_start)
        load_model.clicked.connect(self.model_load)
        self.stop_btn.clicked.connect(self.predict_stop)
        self.stop_btn.setVisible(False)

        self.train_auto = QtGui.QPushButton('Start Auto Training')
        self.train_auto.clicked.connect(self.auto_train)
        self.stop_auto1 = QtGui.QPushButton('Stop Auto Training')
        self.stop_auto1.clicked.connect(self.stop_auto)
        self.stop_auto1.setVisible(False)
        
        grid2 = QtGui.QHBoxLayout()
        grid2.addWidget(self.train_auto)
        grid2.addWidget(self.stop_auto1)
        grid2.addWidget(self.stop_btn)
        grid2.addWidget(self.predict_btn)
        grid2.addWidget(webcam_dev)
        grid2.addWidget(load_model)
        grid2.addWidget(close_btn)

        self.play_widget = QtGui.QWidget()
        self.play_widget.setLayout(grid2)

        grid = QtGui.QGridLayout()
        grid.addWidget(self.view_panel,0,0,1,2)
        grid.addWidget(self.play_widget,1,1)
        grid.addWidget(self.process_lbl,2,0)

        self.setLayout(grid)
        self.show()
       
        self.setWindowTitle("Webcam Interface")
        self.start_cam()

    def out(self):
        if self.mode==1:
            if self.cap.isOpened():
                self.cap.release()

        self.close()

    def save_cropped(self):
        dirName = QtGui.QFileDialog.getExistingDirectory(self, "Choose Directory",
                QtCore.QDir.currentPath())
        if not(dirName):
            return
        else:
            namafile,ok = QtGui.QInputDialog.getText(self,'Input Name','Masukkan Nama',
                                                  QtGui.QLineEdit.Normal)
            if not(ok):
                return

        self.namafile = str(namafile)
        self.save_dirName = str(dirName)
        os.chdir(self.save_dirName)
        self.save_img()
  
    def save_img(self):
        if len(self.data)==0:
            self.process_lbl.setText('No image saved')
            return
        
        for i in range(len(self.data)):
            name_file = self.namafile+'_'+str(self.count)+'.jpg'
            cv2.imwrite(name_file,self.data[i])
            self.count += 1
            self.process_lbl.setText(name_file+' saved')

    def auto_train(self):
        dirName = QtGui.QFileDialog.getExistingDirectory(self, "Choose Directory",
                QtCore.QDir.currentPath())
        if not(dirName):
            return
        else:
            k = os.listdir(str(dirName))
            if len(k)==0:
                namafile,ok = QtGui.QInputDialog.getText(self,'Input Name','Masukkan Nama',
                                                      QtGui.QLineEdit.Normal)
                if not(ok):
                    return

                self.namafile = str(namafile)
            else:
                
                self.namafile = k[0].split('.')[0]

        self.save_dirName = str(dirName)
        os.chdir(self.save_dirName)
            
        self.stop_auto1.setVisible(True)
        self.train_auto.setVisible(False)
        self.timer1 = QtCore.QTimer()
        self.timer1.timeout.connect(self.save_img)
        self.timer1.start(1000.)

    def stop_auto(self):
        self.stop_auto1.setVisible(False)
        self.train_auto.setVisible(True)
        self.timer1.stop()

    def processing_time(self):
        self.process_lbl.setText('Processing Time / Delay : '+ str(self.delayTime) +' s')
        if self.delayTime > 1:
            self.cur_delay = self.delayTime
##            self.timer.stop()
##            self.timer.start(self.cur_delay*1000.)

    def capturing(self):
        start = time()
        QtGui.QApplication.processEvents()
        ret, frame = self.cap.read()
        if ret:
            QtGui.QApplication.processEvents()
            self.img = cv2.resize(frame,(150,150))
            self.detect_face(self.img)
            if self.real_time_predict:
                QtGui.QApplication.processEvents()
                self.predict()
                self.show_on_source()
            else:
                btr = QtGui.QImage(self.img.data, self.img.shape[1],
                                   self.img.shape[0], 3*self.img.shape[1],
                                   QtGui.QImage.Format_RGB888)
                pixmap = self.showImage(btr)
                QtGui.QApplication.processEvents()
                self.view_panel.setPixmap(pixmap)
                
            
            self.delayTime = round(time()- start,3)
            QtGui.QApplication.processEvents()
            self.processing_time()
            
    def start_cam(self):
        self.cap = cv2.VideoCapture(self.current_device)
        if self.cap.isOpened():
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.capturing)
            self.timer.start(self.cur_delay*1000.)
            
        else:
            self.process_lbl.setText('Device not work, please change it')

    def device_change(self):
        f=0
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                f += 1
        cap.release()
        devc,ok = QtGui.QInputDialog.getInt(self,'Webcam Device Channel',
                                              'Pilih webcam source channel :',
                                                self.current_device,0,f-1,1)
        if ok:
            if self.cap.isOpened():
                self.cap.release()
                
            self.current_device = devc
            self.start_cam()
        else:
            self.process_lbl.setText('Device change canceled.')
            return

    def showImage(self,images):
        pixmap = QtGui.QPixmap.fromImage(images)
        pixmap = pixmap.scaled(500,500,QtCore.Qt.KeepAspectRatio,
                               transformMode=QtCore.Qt.SmoothTransformation)
        return pixmap
    
    def image_resizing(self):
        img = self.img
        if img.shape[0]>img.shape[1]:
            shape = img.shape[0]
            r = 640.0 / shape
            newdim = (int(img.shape[1]*r),640)
        else:
            shape = img.shape[1]
            r = 640.0 / shape
            newdim = (640,int(img.shape[0]*r))
        self.img = cv2.resize(img,newdim,interpolation = cv2.INTER_AREA)

    def buka_file(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self, "Open File",
                QtCore.QDir.currentPath())
        if not(fileName):
            return
        fileName = str(fileName)
        self.fileName = fileName
        self.img = cv2.imread(self.fileName)
        if max(self.img.shape) > 800:
            self.image_resizing()
        QtGui.QApplication.processEvents()
        self.detect_face(self.img)
        if self.model is not None:
            self.predict()
        QtGui.QApplication.processEvents()
        self.show_on_source()

    def detect_face(self,img):
        dets = getFaceLoc(img)
        for i in range(len(dets)):
            x = dets[i].left()
            y = dets[i].top()
            w = dets[i].right()- dets[i].left()
            h = dets[i].bottom() - dets[i].top()
            crop = img[y:y+h,x:x+w,:]
            self.rect[i] = [x,y,w,h]

        if len(dets)==0:
            self.rect={}
            
        self.data = get_Face(img,dets,self.imgDim,self.Indices)

    def show_on_source(self):
        image = self.img
        if len(self.rect)!=0:
            for i in range(len(self.rect)):
                QtGui.QApplication.processEvents()
                [x,y,w,h] = self.rect[i]
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
                if self.label_predicted!=None:
                    person = self.label_predicted
                else:
                    person = 'unknown'
                cv2.putText(image,'{}'.format(person),(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,(0,0,255),2)
                
        QtGui.QApplication.processEvents()
        btr = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                           3*image.shape[1], QtGui.QImage.Format_RGB888)
        pixmap = self.showImage(btr)
        QtGui.QApplication.processEvents()
        self.view_panel.setPixmap(pixmap)

    def model_load(self):
        dirName = QtGui.QFileDialog.getExistingDirectory(self, "Choose Directory",
                QtCore.QDir.currentPath())
        if not(dirName):
            return
        dirname = str(dirName)
        err = dirname.split('_')[0].split('\\')[-1]
        if err not in class_build:
            return
        
        self.classf = class_build[err]
        os.chdir(dirname)
        file_list = os.listdir(dirname)
        file_list.sort()
        if self.classf==0:
            QtGui.QApplication.processEvents()
            self.process_lbl.setText('Model load....')
            params = pickle.load(open(file_list[1],'rb'))
            self.num_classes = params[0]
            self.encod = params[1]
##            QtGui.QApplication.processEvents()
##            self.model_to_process = pickle.load(open(file_list[2],'rb'))
            mpath = dirname+'/'+file_list[0]
            QtGui.QApplication.processEvents()
            self.model = load_model(mpath)
        else:
            QtGui.QApplication.processEvents()
            self.process_lbl.setText('Model load....')
            self.model = pickle.load(open(file_list[0],'rb'))
            self.model_to_process = pickle.load(open(file_list[1],'rb'))

        self.process_lbl.setText('Model Loaded.')

    def predict_stop(self):
        self.real_time_predict=False
        self.predict_btn.setVisible(True)
        self.stop_btn.setVisible(False)

    def predict_start(self):
        self.real_time_predict = True
        self.predict_btn.setVisible(True)
        self.stop_btn.setVisible(False)

    def predict(self):
        if self.model is None:
            self.process_lbl.setVisible(True)
            self.process_lbl.setText('No Model Loaded')
            return
        if self.data is not None:
            g = self.classf
            if g==0:
                self.predict_keras()
            else:
                self.predict_sgd()
            QtGui.QApplication.processEvents()
            self.show_on_source()
        else:
            self.label_predicted = None
            self.process_lbl.setText('No face detected')

    def predict_sgd(self):
        imgDim = self.imgDim
        for i in range(len(self.data)):
            gray = cv2.cvtColor(self.data[i], cv2.COLOR_BGR2GRAY)
            image = normalize_image(gray)
            image = cv2.resize(image,(imgDim,imgDim)).flatten()
            image = image.reshape(1,-1)
            pred = self.model.predict(image)
            decs = self.model.decision_function(image)
            maks= max(abs(decs[0]))
            mins = min(abs(decs[0]))
            decs = (decs[0]-mins)/(maks-mins)
            score = max(decs)
            self.process_lbl.setText('Predicted : {} with Score : {}%'.format(str(pred[0]),
                                                                        str(score)))
            self.label_predicted = str(pred)+' '+str(score)
            
    def predict_keras(self):
        imgDim = self.imgDim
        for i in range(len(self.data)):
            gray = cv2.cvtColor(self.data[i], cv2.COLOR_BGR2GRAY)
            image = normalize_image(gray)
            image = cv2.resize(image,(imgDim,imgDim)).flatten()
            image = np.expand_dims(image,axis=0)
            pred = self.model.predict(image)
            pred = get_decode(pred[0],self.encod)
            decs = self.model.predict_proba(image)
            score = max(decs[0])*100
            self.process_lbl.setText('Predicted : {} with Score : {}%'.format(str(pred),
                                                                        str(score)))
            self.label_predicted = str(pred)+' '+str(score)
        
        

if __name__ == '__main__':
    import sys
    m=1
    app = QtGui.QApplication(sys.argv)
    dialog = Window(m)
    dialog.show()

    sys.exit(app.exec_())
            
        
        

        
