from PyQt4 import QtGui,QtCore
from Queue import Queue
import numpy as np
import cv2
import os,sys
from time import sleep,time
import numpy as np
from sklearn.linear_model import SGDClassifier as sgdc
from sklearn.neural_network import MLPClassifier as net
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD,RMSprop,Adagrad,Adadelta,Adam,Adamax,Nadam
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D,ZeroPadding2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


keras_compiler = ['SGD','rmsProp','Adagrad','Adadelta','Adam','Adamax','Nadam']

label_params = {'Zero Padding':'Pad Size','Convolution':'Kernel Size',
                'Max Pooling':'Kernel Size','Flatten':'',
                'Drop Out':'Percentage','Dense':'Neurons'}

loss = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron',
        'squared_loss','huber','epsilon_insensitive','squared_epsilon_insensitive']

learn_type = ['constant','optimal']

mlp_activ = ['identity','logistic','tanh','relu']
mlp_solver = ['lbfgs','sgd','adam']
mlp_learn_type = ['constant','invscalling','adaptive']

def sgd_build(arg):
    if arg[3] == 'optimal':
        k = 0.0
    else:
        k = 0.01
    model = sgdc(loss=arg[0],alpha=arg[1],shuffle=arg[2],learning_rate=arg[3],
                 warm_start=arg[4], average=arg[5],eta0=k)

    return [model,arg[6]]

def mlp_build(arg):
    model = net(hidden_layer_sizes=arg[0],activation=arg[1],solver=arg[2],
                batch_size=arg[3],learning_rate=arg[4],
                learning_rate_init=arg[5],max_iter=arg[6],shuffle=arg[7],
                tol=arg[8],warm_start=arg[9],momentum=arg[10],
                nesterovs_momentum=arg[11],early_stopping=arg[12],
                validation_fraction=arg[13])

    return [model,arg[14]]
            
def optimizer_setup(index,arg):
    if index==0:
        opt = SGD(lr=arg[0],momentum=arg[1],decay=arg[2],nesterov=arg[3])
    elif index==1:
        opt = RMSprop(lr=arg[0],rho=arg[1],epsilon=arg[2],decay=arg[3])
    elif index==2:
        opt = Adagrad(lr=arg[0],epsilon=arg[1],decay=arg[2])
    elif index==3:
        opt = Adadelta(lr=arg[0],rho=arg[1],epsilon=arg[2],decay=arg[3])
    elif index==4:
        opt = Adam(lr=arg[0],beta_1=arg[1],beta_3=arg[2],epsilon=arg[3],
                    decay=arg[4])
    elif index==5:
        opt = Adamax(lr=arg[0],beta_1=arg[1],beta_3=arg[2],epsilon=arg[3],
                    decay=arg[4])
    elif index==6:
        opt = Nadam(lr=arg[0],beta_1=arg[1],beta_3=arg[2],epsilon=arg[3],
                    decay=arg[4])

    return opt

def layer_dev(index,indim,arg,inputDim):
    ind1 = [0,3,4]
    ind2 = [2]
    ind3 = [5]
    ind4 = [1]
    ind5 = [6]
    if index in [0,1,3]:
        inputDim = (inputDim,)
        
    if len(arg)==1 and (index in ind1):
        if index != 0 : index = index-2
        
        if indim:
            if index==2:
                layer_in = None
                return layer_in
            inputDim = int(np.sqrt(inputDim[0]))
            model_set = [ZeroPadding2D((arg[0],arg[0]),input_shape=(1,inputDim,inputDim)),Flatten(input_shape=(1,inputDim,inputDim)),
                     Dropout(arg[0])]
        else:
            model_set = [ZeroPadding2D((arg[0],arg[0])),Flatten(),Dropout(arg[0])]
        layer_in = [model_set[index]]
    elif len(arg)==2 and (index in ind2):
        if indim:
            layer_in = None
            return layer_in
        else:
            model_set = [MaxPooling2D((arg[0],arg[0]),(arg[1],arg[1]))]
        layer_in = model_set
    elif len(arg)==3 and (index in ind3):
        if arg[2] != None:
            arg[2] = maxnorm(arg[2])
        if arg[1] is not None:
            if indim:
                model_set = [Dense(arg[0],activation=arg[1],W_constraint=arg[2],
                                   input_dim=inputDim)]
            else:
                model_set = [Dense(arg[0],activation=arg[1],W_constraint=arg[2])]
            layer_in = model_set
        else:
            if indim:
                model_set = [Dense(arg[0],W_constraint=maxnorm(arg[2]),
                                   input_dim=inputDim)]
            else:
                model_set = [Dense(arg[0],W_constraint=maxnorm(arg[2]))]
            layer_in = model_set
    elif len(arg)==4 and (index in ind4):
        if indim:
            inputDim = int(np.sqrt(inputDim[0]))
            model_set = [Convolution2D(arg[0],arg[1],arg[1],activation=arg[2],name=arg[3],
                                       input_shape=(1,inputDim,inputDim))]
        else:
            model_set = [Convolution2D(arg[0],arg[1],arg[1],activation=arg[2],name=arg[3])]
        layer_in = model_set
    elif len(arg)==1 and (index in ind5):
        if indim:
            layer_in = None
            return layer_in
        else:
            model_set = [Activation(arg[0])]
        layer_in = model_set
        
    return layer_in[0]

def model_dev(index,arg,count,inputDim):                
    if index==5:
        if arg[2]==0:
            arg[2]=None
    if index==0:
        model_str = "(ZeroPadding2D({}))".format(str(arg[0]))
    elif index==1:
        model_str = "(Convolution2D({},{},activation='{}',name='{}'))".format(str(arg[0]),
                                                                str(arg[1]),
                                                   str(arg[2]),str(arg[3]))
    elif index==2:
        model_str = "(MaxPooling2D({},strides={}))".format(str(arg[0]),str(arg[1]))
    elif index==3:
        model_str = '(Flatten())'
    elif index==4:
        model_str = "(Dropout({}))".format(str(arg[0]))
    elif index==6:
        model_str = "(Activation('{}'))".format(str(arg[0]))
    elif index==5:
        if arg[2]==None:
            model_str = "(Dense({},activation='{}',W_constraint=None))".format(str(arg[0]),
                                                                        str(arg[1]),
                                                                        str(arg[2]))
        else:
            model_str = "(Dense({},activation='{}',W_constraint=maxnorm({})))".format(str(arg[0]),
                                                                str(arg[2]))
    
    if count==1:
        str_model = model_str.rsplit(')')
        if len(str_model)==3:
            new = [str_model[0]+',input_dim={}'.format(str(inputDim))+'))']
        elif len(str_model)==4:
            new = [str_model[0]+')'+',input_dim={}'.format(str(inputDim))+'))']
        add_layer = layer_dev(index,True,arg,inputDim)
    else:
        new = [model_str]
        add_layer = layer_dev(index,False,arg,inputDim)

    return [new,add_layer]
        

activ = ['relu','tanh','linear','sigmoid','hard_sigmoid','softsign',
         'softplus','softmax',None]

m=0
class Window(QtGui.QDialog):
    def __init__(self,m,n):
        QtGui.QDialog.__init__(self)
        self.m = m
        self.num_classes = n
        self.init_model()
        self.setWindowIcon(QtGui.QIcon(os.getcwd()+'/Oxygen.ico'))
        
    def init_model(self):
        if self.m ==0:
            self.kerasUI()
            self.count = 0
            self.current_layers = []
            self.conv_nama =[]
            self.keras_model_build= None
            self.last_layer_arg = None
            self.last_layer_err = 0
            self.ind = 0
            self.string_layers = []
        elif self.m == 1:
            self.sgdUI()
            self.sgd_model_build = None
        elif self.m == 2:
            self.mlpUI()
            self.mlp_hlayer = ()
            self.mlp_model_build = None
        else:
            return
            
    def sgdUI(self):
        sgd_g1 = QtGui.QGroupBox('Learning Parameters')
        sgd_g1.setAlignment(QtCore.Qt.AlignCenter)
        close_button = QtGui.QPushButton('Close')
        close_button.clicked.connect(self.close)

        alpha_lbl = QtGui.QLabel('Learning Rate')
        self.alpha_p1 = QtGui.QDoubleSpinBox()
        self.alpha_p1.setRange(0,1)
        self.alpha_p1.setSingleStep(0.001)
        self.alpha_p1.setValue(0.01)
        VBox1 = QtGui.QVBoxLayout()
        VBox1.addWidget(alpha_lbl)
        VBox1.addWidget(self.alpha_p1)

        learn_method_lbl = QtGui.QLabel('Learning Type')
        self.CBox1 = QtGui.QComboBox()
        self.CBox1.addItem('constant',0)
        self.CBox1.addItem('optimal',1)
        VBox2 = QtGui.QVBoxLayout()
        VBox2.addWidget(learn_method_lbl)
        VBox2.addWidget(self.CBox1)

        HBox1 = QtGui.QHBoxLayout()
        HBox1.addLayout(VBox1)
        HBox1.addLayout(VBox2)
        sgd_g1.setLayout(HBox1)

        sgd_g2 = QtGui.QGroupBox('Compile Parameters')
        loss_lbl = QtGui.QLabel('Loss Function')
        self.CBox2 = QtGui.QComboBox()
        for i in range(len(loss)):
            self.CBox2.addItem(loss[i],i)

        self.shuffle = QtGui.QCheckBox('Shuffle data')
        self.shuffle.setChecked(True)
        self.warm_start = QtGui.QCheckBox('Using Precious Call Fit')
        self.warm_start.setChecked(False)
        VBox5 = QtGui.QVBoxLayout()
        VBox5.addWidget(self.shuffle)
        VBox5.addWidget(self.warm_start)
        self.sgd_g4 = QtGui.QGroupBox('Additional Parameter')
        self.sgd_g4.setCheckable(True)
        self.sgd_g4.setChecked(False)
        self.sgd_g4.setLayout(VBox5)
        
        self.average = QtGui.QGroupBox('Averaging Wights')
        average_lbl = QtGui.QLabel('Averaging every n samples: ')
        self.average_p1 = QtGui.QSpinBox()
        self.average_p1.setFixedWidth(50)
        self.average_p1.setRange(0,1000)
        self.average_p1.setValue(1)
        self.average.setCheckable(True)
        self.average.setChecked(False)
        ave = QtGui.QVBoxLayout()
        ave.addWidget(average_lbl)
        ave.addWidget(self.average_p1)
        self.average.setLayout(ave)

        VBox3 = QtGui.QVBoxLayout()
        VBox3.addWidget(loss_lbl)
        VBox3.addWidget(self.CBox2)

        epoch_lbl = QtGui.QLabel('Number of Epoch(iterasi): ')
        self.epoch_p1 = QtGui.QSpinBox()
        self.epoch_p1.setFixedWidth(50)
        self.epoch_p1.setRange(0,200)
        self.epoch_p1.setValue(10)
        VBox4 = QtGui.QVBoxLayout()
        VBox4.addWidget(epoch_lbl)
        VBox4.addWidget(self.epoch_p1)

        HBox2 = QtGui.QHBoxLayout()
        HBox2.addLayout(VBox3)
        HBox2.addLayout(VBox4)
        sgd_g2.setLayout(HBox2)

        save_model = QtGui.QPushButton('Save Model')
        save_model.clicked.connect(self.sgd_params)
        save_model.setFixedWidth(100)
        self.status = QtGui.QLabel('')
        sgd_g3 = QtGui.QGroupBox()
        HBox3 = QtGui.QHBoxLayout()
        HBox3.addWidget(self.status)
        HBox3.addWidget(save_model)
        HBox3.addWidget(close_button)
        sgd_g3.setLayout(HBox3)
        
        mainLayout = QtGui.QGridLayout()
        mainLayout.addWidget(sgd_g1,0,0)
        mainLayout.addWidget(sgd_g2,0,1)
        mainLayout.addWidget(self.average,1,0)
        mainLayout.addWidget(sgd_g3,1,1)
        mainLayout.addWidget(self.sgd_g4,2,0)

        self.setLayout(mainLayout)
        self.setWindowTitle("SGDClassifier Interface")

    def sgd_params(self):
        p2 = self.alpha_p1.value()
        p4 = learn_type[self.CBox1.currentIndex()]
        p1 = loss[self.CBox2.currentIndex()]
        p7 = self.epoch_p1.value()
        if self.average.isChecked():
            p6 = self.average_p1.value()
        else:
            p6 = False

        if self.sgd_g4.isChecked():
            p3 = self.shuffle.isChecked()
            p5 = self.warm_start.isChecked()
        else:
            p3 = True
            p5 = False
            
        self.status.setText('Compiling Model Done')
        arg = [p1,p2,p3,p4,p5,p6,p7]
        self.sgd_model_build = sgd_build(arg)

    def mlpUI(self):
        mlp_g1 = QtGui.QGroupBox('Learning Parameters')
        mlp_g1.setAlignment(QtCore.Qt.AlignCenter)
        close_button = QtGui.QPushButton('Close')
        close_button.clicked.connect(self.close)

        hidden1_lbl = QtGui.QLabel('Neurons')
        self.hidden_p2 = QtGui.QSpinBox()
        self.hidden_p2.setRange(0,10000)
        self.hidden_p2.setValue(768)
        hidden_btn = QtGui.QPushButton('Add Hidden\nLayer')
        hidden_btn.clicked.connect(self.set_mlp_layer)
        self.hidden2_lbl = QtGui.QLabel('')
        grid1 = QtGui.QGridLayout()
        grid1.addWidget(hidden1_lbl,0,0,1,2)
        grid1.addWidget(self.hidden_p2,1,0,1,2)
        grid1.addWidget(self.hidden2_lbl,2,0)
        grid1.addWidget(hidden_btn,2,2)
        mlp_g1.setLayout(grid1)

        activ_lbl = QtGui.QLabel('Activation')
        solver_lbl = QtGui.QLabel('Solver')
        batch_lbl = QtGui.QLabel('Batch Size')
        learnt_lbl = QtGui.QLabel('Learn Type')
        learn_lbl = QtGui.QLabel('Learn Rate')
        iter_lbl = QtGui.QLabel('Max Iteration')
        tol_lbl = QtGui.QLabel('Loss Tolerance')
        momen_lbl = QtGui.QLabel('SGD Momentum')
        valfrac_lbl = QtGui.QLabel('Validation Fraction')

        self.activ_p1 = QtGui.QComboBox()
        for i in range(len(mlp_activ)):
            self.activ_p1.addItem(mlp_activ[i],i)
        self.activ_p1.setFixedWidth(60)
        
        self.solver_p1 = QtGui.QComboBox()
        for i in range(len(mlp_solver)):
            self.solver_p1.addItem(mlp_solver[i],i)
        self.solver_p1.setFixedWidth(60)

        self.batch_p1 = QtGui.QSpinBox()
        self.batch_p1.setFixedWidth(50)
        self.batch_p1.setRange(0,256)
        self.batch_p1.setSingleStep(32)
        self.batch_p1.setValue(64)

        self.mlp_learn_p1 = QtGui.QComboBox()
        for i in range(len(mlp_learn_type)):
            self.mlp_learn_p1.addItem(mlp_learn_type[i],i)

        self.mlp_learn_p2 = QtGui.QDoubleSpinBox()
        self.mlp_learn_p2.setDecimals(4)
        self.mlp_learn_p2.setRange(0,1)
        self.mlp_learn_p2.setSingleStep(0.0001)
        self.mlp_learn_p2.setValue(0.001)

        self.max_iter = QtGui.QSpinBox()
        self.max_iter.setRange(0,200)
        self.max_iter.setValue(1)
        self.max_iter.valueChanged.connect(self.iter_rule)

        self.tol_p1 = QtGui.QDoubleSpinBox()
        self.tol_p1.setFixedWidth(80)
        self.tol_p1.setDecimals(6)
        self.tol_p1.setRange(0,1)
        self.tol_p1.setSingleStep(0.000001)
        self.tol_p1.setValue(0.0001)

        self.sgd_momen = QtGui.QDoubleSpinBox()
        self.sgd_momen.setRange(0,1)
        self.sgd_momen.setSingleStep(0.01)
        self.sgd_momen.setValue(0.9)

        self.valfrac = QtGui.QDoubleSpinBox()
        self.valfrac.setRange(0,1)
        self.valfrac.setSingleStep(0.01)
        self.valfrac.setValue(0.1)

        grid2 = QtGui.QGridLayout()
        grid2.addWidget(activ_lbl,0,0)
        grid2.addWidget(self.activ_p1,1,0)
        grid2.addWidget(solver_lbl,0,1)
        grid2.addWidget(self.solver_p1,1,1)
        grid2.addWidget(batch_lbl,0,2)
        grid2.addWidget(self.batch_p1,1,2)
        grid2.addWidget(learnt_lbl,2,0)
        grid2.addWidget(self.mlp_learn_p1,3,0)
        grid2.addWidget(learn_lbl,2,1)
        grid2.addWidget(self.mlp_learn_p2,3,1)
        grid2.addWidget(iter_lbl,2,2)
        grid2.addWidget(self.max_iter,3,2)
        mlp_g2 = QtGui.QGroupBox()
        mlp_g2.setLayout(grid2)

        VBox1 = QtGui.QVBoxLayout()
        VBox1.addWidget(tol_lbl)
        VBox1.addWidget(self.tol_p1)
        self.mlp_g3 = QtGui.QGroupBox('Loss Tolerance')
        self.mlp_g3.setCheckable(True)
        self.mlp_g3.setChecked(False)
        self.mlp_g3.setLayout(VBox1)

        VBox2 = QtGui.QVBoxLayout()
        VBox2.addWidget(valfrac_lbl)
        VBox2.addWidget(self.valfrac)
        self.mlp_g4 = QtGui.QGroupBox('Early Stopping')
        self.mlp_g4.setCheckable(True)
        self.mlp_g4.setChecked(False)
        self.mlp_g4.setLayout(VBox2)

        self.neste_momen = QtGui.QCheckBox('Nesterov Momentum')
        self.neste_momen.setChecked(True)
        grid3 = QtGui.QGridLayout()
        grid3.addWidget(momen_lbl,0,0)
        grid3.addWidget(self.sgd_momen,1,0)
        grid3.addWidget(self.neste_momen,0,1)
        self.mlp_g5 = QtGui.QGroupBox('SGD Momentum')
        self.mlp_g5.setCheckable(True)
        self.mlp_g5.setChecked(False)
        self.mlp_g5.setLayout(grid3)

        self.shuffle = QtGui.QCheckBox('Shuffle data')
        self.shuffle.setChecked(True)
        self.warm_start = QtGui.QCheckBox('Using Precious Call Fit')
        self.warm_start.setChecked(False)
        VBox5 = QtGui.QVBoxLayout()
        VBox5.addWidget(self.shuffle)
        VBox5.addWidget(self.warm_start)
        self.sgd_g4 = QtGui.QGroupBox('Additional Parameter')
        self.sgd_g4.setCheckable(True)
        self.sgd_g4.setChecked(False)
        self.sgd_g4.setLayout(VBox5)

        save_model = QtGui.QPushButton('Save Model')
        save_model.clicked.connect(self.mlp_params)
        save_model.setFixedWidth(100)
        self.status = QtGui.QLabel('')
        sgd_g3 = QtGui.QGroupBox()
        HBox3 = QtGui.QHBoxLayout()
        HBox3.addWidget(save_model)
        HBox3.addWidget(close_button)
        sgd_g3.setLayout(HBox3)

        epoch_lbl = QtGui.QLabel('Number of epoch: ')
        self.epoch_p1 = QtGui.QSpinBox()
        self.epoch_p1.setFixedWidth(50)
        self.epoch_p1.setRange(0,200)
        self.epoch_p1.setValue(1)
        VBox6 = QtGui.QVBoxLayout()
        VBox6.addWidget(epoch_lbl)
        VBox6.addWidget(self.epoch_p1)
        mlp_g6 = QtGui.QGroupBox('Epoch Setting')
        mlp_g6.setLayout(VBox6)
        
        mainLayout = QtGui.QGridLayout()
        mainLayout.addWidget(mlp_g1,0,0)
        mainLayout.addWidget(mlp_g2,0,1,1,2)
        mainLayout.addWidget(self.mlp_g3,1,0)
        mainLayout.addWidget(self.mlp_g4,1,1)
        mainLayout.addWidget(self.mlp_g5,1,2)
        mainLayout.addWidget(self.sgd_g4,2,0)
        mainLayout.addWidget(mlp_g6,2,1)
        mainLayout.addWidget(sgd_g3,2,2)
        mainLayout.addWidget(self.status,3,0)

        self.setLayout(mainLayout)
        self.setWindowTitle("MLP Perceptron Interface")

    def set_mlp_layer(self):
        if len(self.mlp_hlayer)>10:
            self.status.setText('Hidden Layer too much')
            return
        self.mlp_hlayer = self.mlp_hlayer + (self.hidden_p2.value(),)
        self.hidden2_lbl.setText(str(self.mlp_hlayer))
        
    def iter_rule(self,v):
        maks = int(round((200.0/v))-1)
        if maks==0:
            maks = 1
        self.epoch_p1.setRange(0,maks)
        self.epoch_p1.setValue(1)

    def mlp_params(self):
        if len(self.mlp_hlayer)==0: p1=(100,)
        else : p1 = self.mlp_hlayer

        p2 = mlp_activ[self.activ_p1.currentIndex()]
        p3 = mlp_solver[self.solver_p1.currentIndex()]
        
        if self.batch_p1.value()==0: p4='auto'
        else: p4 = self.batch_p1.value()
        
        p5 = mlp_learn_type[self.mlp_learn_p1.currentIndex()]
        p6 = round(self.mlp_learn_p2.value(),4)
        p7 = self.max_iter.value()
        if self.sgd_g4.isChecked():
            p8 = self.shuffle.isChecked()
            p10 = self.warm_start.isChecked()
        else:
            p8 = True
            p10 = False

        if self.mlp_g3.isChecked(): p9 = round(self.tol_p1.value(),6)
        else : p9 = 0.0001

        if self.mlp_g5.isChecked():
            p11 = self.sgd_momen.value()
            p12 = self.neste_momen.isChecked()
        else:
            p11 = 0.9
            p12 = True

        if self.mlp_g4.isChecked():
            p13 = True
            p14 = self.valfrac.value()
        else:
            p13 = False
            p14 = 0.1

        p15 = self.epoch_p1.value()
        self.status.setText('Compiling Model Done')
        arg = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15]
        self.mlp_model_build = mlp_build(arg)
        del arg
        
    def kerasUI(self):

        sgd_vb1 = QtGui.QVBoxLayout()
        sgd_vb2 = QtGui.QVBoxLayout()
        sgd_hb1 = QtGui.QHBoxLayout()

        sgd_lr_lbl = QtGui.QLabel('Learning rate')
        sgd_mn_lbl = QtGui.QLabel('Momentum')
        sgd_dc_lbl = QtGui.QLabel('Decay')
        self.sgd_nest = QtGui.QCheckBox('Nesterov Moementum')
        self.sgd_nest.setChecked(True)
        self.sgd_lr = QtGui.QDoubleSpinBox()
        self.sgd_lr.setDecimals(4)
        self.sgd_lr.setRange(0,1)
        self.sgd_lr.setSingleStep(0.001)
        self.sgd_lr.setValue(0.01)
        
        self.sgd_mn = QtGui.QDoubleSpinBox()
        self.sgd_mn.setDecimals(2)
        self.sgd_mn.setRange(0,1)
        self.sgd_mn.setSingleStep(0.01)
        self.sgd_mn.setValue(0.0)

        self.sgd_dc = QtGui.QDoubleSpinBox()
        self.sgd_dc.setDecimals(7)
        self.sgd_dc.setRange(0,1)
        self.sgd_dc.setSingleStep(1e-6)
        self.sgd_dc.setValue(0.0)
        
        sgd_vb1.addWidget(sgd_lr_lbl)
        sgd_vb1.addWidget(self.sgd_lr)
        sgd_vb1.addWidget(sgd_mn_lbl)
        sgd_vb1.addWidget(self.sgd_mn)
        sgd_vb2.addWidget(sgd_dc_lbl)
        sgd_vb2.addWidget(self.sgd_dc)
        sgd_vb2.addWidget(self.sgd_nest)
        sgd_hb1.addLayout(sgd_vb1)
        sgd_hb1.addLayout(sgd_vb2)
        self.sgd_widget = QtGui.QWidget()

        rms_vb1 = QtGui.QVBoxLayout()
        rms_vb2 = QtGui.QVBoxLayout()
        rms_hb1 = QtGui.QHBoxLayout()

        rms_lr_lbl = QtGui.QLabel('Learning rate')
        rms_rho_lbl = QtGui.QLabel('Rho')
        rms_eps_lbl = QtGui.QLabel('Epsilon')
        rms_dc_lbl = QtGui.QLabel('Decay')
        
        self.rms_lr = QtGui.QDoubleSpinBox()
        self.rms_lr.setDecimals(4)
        self.rms_lr.setRange(0,1)
        self.rms_lr.setSingleStep(0.0001)
        self.rms_lr.setValue(0.001)
        
        self.rms_rho = QtGui.QDoubleSpinBox()
        self.rms_rho.setDecimals(2)
        self.rms_rho.setRange(0,1)
        self.rms_rho.setSingleStep(0.01)
        self.rms_rho.setValue(0.9)

        self.rms_dc = QtGui.QDoubleSpinBox()
        self.rms_dc.setDecimals(7)
        self.rms_dc.setRange(0,1)
        self.rms_dc.setSingleStep(1e-6)
        self.rms_dc.setValue(0.0)
        
        self.rms_eps = QtGui.QDoubleSpinBox()
        self.rms_eps.setDecimals(9)
        self.rms_eps.setRange(0,1)
        self.rms_eps.setSingleStep(1e-8)
        self.rms_eps.setValue(1e-8)

        rms_vb1.addWidget(rms_lr_lbl)
        rms_vb1.addWidget(self.rms_lr)
        rms_vb1.addWidget(rms_rho_lbl)
        rms_vb1.addWidget(self.rms_rho)
        rms_vb2.addWidget(rms_eps_lbl)
        rms_vb2.addWidget(self.rms_eps)
        rms_vb2.addWidget(rms_dc_lbl)
        rms_vb2.addWidget(self.rms_dc)
        rms_hb1.addLayout(rms_vb1)
        rms_hb1.addLayout(rms_vb2)
        self.rms_widget = QtGui.QWidget()

        adag_vb1 = QtGui.QVBoxLayout()
        adag_vb2 = QtGui.QVBoxLayout()
        adag_hb1 = QtGui.QHBoxLayout()

        adag_lr_lbl = QtGui.QLabel('Learning rate')
        adag_eps_lbl = QtGui.QLabel('Epsilon')
        adag_dc_lbl = QtGui.QLabel('Decay')
        
        self.adag_lr = QtGui.QDoubleSpinBox()
        self.adag_lr.setDecimals(3)
        self.adag_lr.setRange(0,1)
        self.adag_lr.setSingleStep(0.001)
        self.adag_lr.setValue(0.01)
      
        self.adag_eps = QtGui.QDoubleSpinBox()
        self.adag_eps.setDecimals(7)
        self.adag_eps.setRange(0,1)
        self.adag_eps.setSingleStep(1e-6)
        self.adag_eps.setValue(0.0)

        self.adag_dc = QtGui.QDoubleSpinBox()
        self.adag_dc.setDecimals(9)
        self.adag_dc.setRange(0,1)
        self.adag_dc.setSingleStep(1e-8)
        self.adag_dc.setValue(1e-8)

        adag_vb1.addWidget(adag_lr_lbl)
        adag_vb1.addWidget(self.adag_lr)
        adag_vb1.addWidget(adag_eps_lbl)
        adag_vb1.addWidget(self.adag_eps)
        adag_vb2.addWidget(adag_dc_lbl)
        adag_vb2.addWidget(self.adag_dc)
        adag_hb1.addLayout(adag_vb1)
        adag_hb1.addLayout(adag_vb2)
        self.adag_widget = QtGui.QWidget()

        adad_vb1 = QtGui.QVBoxLayout()
        adad_vb2 = QtGui.QVBoxLayout()
        adad_hb1 = QtGui.QHBoxLayout()

        adad_lr_lbl = QtGui.QLabel('Learning rate')
        adad_rho_lbl = QtGui.QLabel('Rho')
        adad_eps_lbl = QtGui.QLabel('Epsilon')
        adad_dc_lbl = QtGui.QLabel('Decay')
        
        self.adad_lr = QtGui.QDoubleSpinBox()
        self.adad_lr.setDecimals(4)
        self.adad_lr.setRange(0,1)
        self.adad_lr.setSingleStep(0.0001)
        self.adad_lr.setValue(0.001)
        
        self.adad_rho = QtGui.QDoubleSpinBox()
        self.adad_rho.setDecimals(2)
        self.adad_rho.setRange(0,1)
        self.adad_rho.setSingleStep(0.01)
        self.adad_rho.setValue(0.9)

        self.adad_dc = QtGui.QDoubleSpinBox()
        self.adad_dc.setDecimals(6)
        self.adad_dc.setRange(0,1)
        self.adad_dc.setSingleStep(1e-6)
        self.adad_dc.setValue(0.0)
        
        self.adad_eps = QtGui.QDoubleSpinBox()
        self.adad_eps.setDecimals(9)
        self.adad_eps.setRange(0,1)
        self.adad_eps.setSingleStep(1e-8)
        self.adad_eps.setValue(1e-8)

        adad_vb1.addWidget(adad_lr_lbl)
        adad_vb1.addWidget(self.adad_lr)
        adad_vb1.addWidget(adad_rho_lbl)
        adad_vb1.addWidget(self.adad_rho)
        adad_vb2.addWidget(adad_eps_lbl)
        adad_vb2.addWidget(self.adad_eps)
        adad_vb2.addWidget(adad_dc_lbl)
        adad_vb2.addWidget(self.adad_dc)
        adad_hb1.addLayout(adad_vb1)
        adad_hb1.addLayout(adad_vb2)
        self.adad_widget = QtGui.QWidget()

        adam_vb1 = QtGui.QVBoxLayout()
        adam_vb2 = QtGui.QVBoxLayout()
        adam_hb1 = QtGui.QHBoxLayout()

        adam_lr_lbl = QtGui.QLabel('Learning rate')
        adam_beta1_lbl = QtGui.QLabel('Beta_1')
        adam_eps_lbl = QtGui.QLabel('Epsilon')
        adam_beta2_lbl = QtGui.QLabel('Beta_2')
        adam_dc_lbl = QtGui.QLabel('Decay')
        
        self.adam_lr = QtGui.QDoubleSpinBox()
        self.adam_lr.setDecimals(4)
        self.adam_lr.setRange(0,1)
        self.adam_lr.setSingleStep(0.0001)
        self.adam_lr.setValue(0.001)
        
        self.adam_beta1 = QtGui.QDoubleSpinBox()
        self.adam_beta1.setDecimals(2)
        self.adam_beta1.setRange(0,1)
        self.adam_beta1.setSingleStep(0.01)
        self.adam_beta1.setValue(0.9)

        self.adam_beta2 = QtGui.QDoubleSpinBox()
        self.adam_beta2.setDecimals(4)
        self.adam_beta2.setRange(0,1)
        self.adam_beta2.setSingleStep(0.0001)
        self.adam_beta2.setValue(0.999)

        self.adam_dc = QtGui.QDoubleSpinBox()
        self.adam_dc.setDecimals(6)
        self.adam_dc.setRange(0,1)
        self.adam_dc.setSingleStep(1e-6)
        self.adam_dc.setValue(0.0)
        
        self.adam_eps = QtGui.QDoubleSpinBox()
        self.adam_eps.setDecimals(9)
        self.adam_eps.setRange(0,1)
        self.adam_eps.setSingleStep(1e-8)
        self.adam_eps.setValue(1e-8)

        adam_vb1.addWidget(adam_lr_lbl)
        adam_vb1.addWidget(self.adam_lr)
        adam_vb1.addWidget(adam_beta1_lbl)
        adam_vb1.addWidget(self.adam_beta1)
        adam_vb1.addWidget(adam_beta2_lbl)
        adam_vb1.addWidget(self.adam_beta2)
        adam_vb2.addWidget(adam_eps_lbl)
        adam_vb2.addWidget(self.adam_eps)
        adam_vb2.addWidget(adam_dc_lbl)
        adam_vb2.addWidget(self.adam_dc)
        adam_hb1.addLayout(adam_vb1)
        adam_hb1.addLayout(adam_vb2)
        self.adam_widget = QtGui.QWidget()

        adamax_vb1 = QtGui.QVBoxLayout()
        adamax_vb2 = QtGui.QVBoxLayout()
        adamax_hb1 = QtGui.QHBoxLayout()

        adamax_lr_lbl = QtGui.QLabel('Learning rate')
        adamax_beta1_lbl = QtGui.QLabel('Beta_1')
        adamax_eps_lbl = QtGui.QLabel('Epsilon')
        adamax_beta2_lbl = QtGui.QLabel('Beta_2')
        adamax_dc_lbl = QtGui.QLabel('Decay')
        
        self.adamax_lr = QtGui.QDoubleSpinBox()
        self.adamax_lr.setDecimals(3)
        self.adamax_lr.setRange(0,1)
        self.adamax_lr.setSingleStep(0.001)
        self.adamax_lr.setValue(0.02)
        
        self.adamax_beta1 = QtGui.QDoubleSpinBox()
        self.adamax_beta1.setDecimals(2)
        self.adamax_beta1.setRange(0,1)
        self.adamax_beta1.setSingleStep(0.01)
        self.adamax_beta1.setValue(0.9)

        self.adamax_beta2 = QtGui.QDoubleSpinBox()
        self.adamax_beta2.setDecimals(4)
        self.adamax_beta2.setRange(0,1)
        self.adamax_beta2.setSingleStep(0.0001)
        self.adamax_beta2.setValue(0.999)

        self.adamax_dc = QtGui.QDoubleSpinBox()
        self.adamax_dc.setDecimals(6)
        self.adamax_dc.setRange(0,1)
        self.adamax_dc.setSingleStep(1e-6)
        self.adamax_dc.setValue(0.0)
        
        self.adamax_eps = QtGui.QDoubleSpinBox()
        self.adamax_eps.setDecimals(9)
        self.adamax_eps.setRange(0,1)
        self.adamax_eps.setSingleStep(1e-8)
        self.adamax_eps.setValue(1e-8)

        adamax_vb1.addWidget(adamax_lr_lbl)
        adamax_vb1.addWidget(self.adamax_lr)
        adamax_vb1.addWidget(adamax_beta1_lbl)
        adamax_vb1.addWidget(self.adamax_beta1)
        adamax_vb1.addWidget(adamax_beta2_lbl)
        adamax_vb1.addWidget(self.adamax_beta2)
        adamax_vb2.addWidget(adamax_eps_lbl)
        adamax_vb2.addWidget(self.adamax_eps)
        adamax_vb2.addWidget(adamax_dc_lbl)
        adamax_vb2.addWidget(self.adamax_dc)
        adamax_hb1.addLayout(adamax_vb1)
        adamax_hb1.addLayout(adamax_vb2)
        self.adamax_widget = QtGui.QWidget()

        nadam_vb1 = QtGui.QVBoxLayout()
        nadam_vb2 = QtGui.QVBoxLayout()
        nadam_hb1 = QtGui.QHBoxLayout()

        nadam_lr_lbl = QtGui.QLabel('Learning rate')
        nadam_beta1_lbl = QtGui.QLabel('Beta_1')
        nadam_eps_lbl = QtGui.QLabel('Epsilon')
        nadam_beta2_lbl = QtGui.QLabel('Beta_2')
        nadam_dc_lbl = QtGui.QLabel('Decay')
        
        self.nadam_lr = QtGui.QDoubleSpinBox()
        self.nadam_lr.setDecimals(3)
        self.nadam_lr.setRange(0,1)
        self.nadam_lr.setSingleStep(0.001)
        self.nadam_lr.setValue(0.02)
        
        self.nadam_beta1 = QtGui.QDoubleSpinBox()
        self.nadam_beta1.setDecimals(2)
        self.nadam_beta1.setRange(0,1)
        self.nadam_beta1.setSingleStep(0.01)
        self.nadam_beta1.setValue(0.9)

        self.nadam_beta2 = QtGui.QDoubleSpinBox()
        self.nadam_beta2.setDecimals(4)
        self.nadam_beta2.setRange(0,1)
        self.nadam_beta2.setSingleStep(0.0001)
        self.nadam_beta2.setValue(0.999)

        self.nadam_dc = QtGui.QDoubleSpinBox()
        self.nadam_dc.setDecimals(6)
        self.nadam_dc.setRange(0,1)
        self.nadam_dc.setSingleStep(1e-6)
        self.nadam_dc.setValue(0.0)
        
        self.nadam_eps = QtGui.QDoubleSpinBox()
        self.nadam_eps.setDecimals(9)
        self.nadam_eps.setRange(0,1)
        self.nadam_eps.setSingleStep(1e-8)
        self.nadam_eps.setValue(1e-8)

        nadam_vb1.addWidget(nadam_lr_lbl)
        nadam_vb1.addWidget(self.nadam_lr)
        nadam_vb1.addWidget(nadam_beta1_lbl)
        nadam_vb1.addWidget(self.nadam_beta1)
        nadam_vb1.addWidget(nadam_beta2_lbl)
        nadam_vb1.addWidget(self.nadam_beta2)
        nadam_vb2.addWidget(nadam_eps_lbl)
        nadam_vb2.addWidget(self.nadam_eps)
        nadam_vb2.addWidget(nadam_dc_lbl)
        nadam_vb2.addWidget(self.nadam_dc)
        nadam_hb1.addLayout(nadam_vb1)
        nadam_hb1.addLayout(nadam_vb2)
        self.nadam_widget = QtGui.QWidget()

        k = [sgd_vb1,sgd_vb2,rms_vb1,rms_vb2,adag_vb1,adag_vb2,
             adad_vb1,adad_vb2,adam_vb1,adam_vb2,adamax_vb1,adamax_vb2,
             nadam_vb1,nadam_vb2]
        for w in k:
            w.setAlignment(QtCore.Qt.AlignCenter)
        del k
            
        self.sgd_widget.setLayout(sgd_hb1)
        self.rms_widget.setLayout(rms_hb1)
        self.adag_widget.setLayout(adag_hb1)
        self.adad_widget.setLayout(adad_hb1)
        self.adam_widget.setLayout(adam_hb1)
        self.adamax_widget.setLayout(adamax_hb1)
        self.nadam_widget.setLayout(nadam_hb1)

        self.sgd_widget.setVisible(True)
        self.rms_widget.setVisible(False)
        self.adag_widget.setVisible(False)
        self.adad_widget.setVisible(False)
        self.adam_widget.setVisible(False)
        self.adamax_widget.setVisible(False)
        self.nadam_widget.setVisible(False)

        self.optimizer_widget = [self.sgd_widget,self.rms_widget,self.adag_widget,
                                 self.adad_widget,self.adam_widget,
                                 self.adamax_widget,self.nadam_widget]

        self.comp_cbox = QtGui.QComboBox()
        for i in range(len(keras_compiler)):
            self.comp_cbox.addItem(keras_compiler[i],i)
        self.comp_cbox.currentIndexChanged.connect(self.optimizer_comp)
        self.comp_cbox.setFixedWidth(80)

        epo_lbl = QtGui.QLabel('Number of epoch')
        self.epo_p1 = QtGui.QSpinBox()
        self.epo_p1.setValue(1)
        self.epo_p1.setFixedWidth(40)

        bs_lbl = QtGui.QLabel('Batch Size')
        self.bs_p1 = QtGui.QSpinBox()
        self.bs_p1.setRange(0,256)
        self.bs_p1.setValue(64)
        self.bs_p1.setFixedWidth(50)
        
        grid_comp = QtGui.QGridLayout()
        grid_comp.addWidget(self.comp_cbox,0,0)
        grid_comp.addWidget(epo_lbl,1,0)
        grid_comp.addWidget(self.epo_p1,2,0)
        grid_comp.addWidget(bs_lbl,3,0)
        grid_comp.addWidget(self.bs_p1,4,0)
        grid_comp.addWidget(self.sgd_widget,0,1)
        grid_comp.addWidget(self.rms_widget,0,1)
        grid_comp.addWidget(self.adag_widget,0,1)
        grid_comp.addWidget(self.adad_widget,0,1)
        grid_comp.addWidget(self.adam_widget,0,1)
        grid_comp.addWidget(self.adamax_widget,0,1)
        grid_comp.addWidget(self.nadam_widget,0,1)

        opt_gbox = QtGui.QGroupBox('Optimizer Setup')
        opt_gbox.setLayout(grid_comp)

                
        keras_g1 = QtGui.QGroupBox('Input Dimension')
        keras_g1.setAlignment(QtCore.Qt.AlignCenter)
        close_button = QtGui.QPushButton('Close')
        close_button.clicked.connect(self.close)

        HBox1_1 = QtGui.QHBoxLayout()
        self.dim = QtGui.QSpinBox()
        self.dim.setRange(0,350)
        self.dim.setValue(150)
        dim_btn = QtGui.QPushButton('Set \nInput Dim')
        dim_btn.clicked.connect(self.input_dim)
        HBox1_1.addWidget(self.dim)
        HBox1_1.addWidget(dim_btn)

        HBox1_2 = QtGui.QHBoxLayout()
        self.dim_lbl = QtGui.QLabel('')
        HBox1_2.addWidget(self.dim_lbl)
        HBox1_2.setAlignment(QtCore.Qt.AlignCenter)

        k = self.dim.value()
        self.dim_lbl.setText('{}x{} = {}'.format(str(k),str(k),str(k*k)))
        self.inputDim = self.dim.value()*self.dim.value()
        
        VBox1 = QtGui.QVBoxLayout()
        VBox1.addLayout(HBox1_1)
        VBox1.addLayout(HBox1_2)
        keras_g1.setLayout(VBox1)

        keras_g2 = QtGui.QGroupBox('Select Layer Configuration')
        keras_g2.setAlignment(QtCore.Qt.AlignCenter)
        VBox1 = QtGui.QVBoxLayout()
        self.CBox1 = QtGui.QComboBox()
        self.CBox1.addItem('Zero Padding',0)
        self.CBox1.addItem('Convolution',1)
        self.CBox1.addItem('Max Pooling',2)
        self.CBox1.addItem('Flatten',3)
        self.CBox1.addItem('Drop Out',4)
        self.CBox1.addItem('Dense',5)
        self.CBox1.addItem('Activation',6)
        self.CBox1.currentIndexChanged.connect(self.set_params)
        VBox1.addWidget(self.CBox1)
        keras_g2.setLayout(VBox1)

        VBox2_1 = QtGui.QVBoxLayout()
        zeropad_lbl = QtGui.QLabel('Pad Size')
        self.zeropad_p1 = QtGui.QSpinBox()
        self.zeropad_p1.setValue(1)
        VBox2_1.addWidget(zeropad_lbl)
        VBox2_1.addWidget(self.zeropad_p1)
        VBox2_1.setAlignment(QtCore.Qt.AlignCenter)

        VBox2_2 = QtGui.QVBoxLayout()
        conv_lbl_1 = QtGui.QLabel('Number of Filters :')
        conv_lbl_2 = QtGui.QLabel('Filter Length')
        conv_lbl_3 = QtGui.QLabel('Activation')
        conv_lbl_4 = QtGui.QLabel('Name')
        self.conv_p1 = QtGui.QSpinBox()
        self.conv_p1.setRange(0,10000)
        self.conv_p1.setValue(64)
        self.conv_p2 = QtGui.QSpinBox()
        self.conv_p2.setRange(0,1000)
        self.conv_p2.setValue(3)
        self.conv_p3 = QtGui.QComboBox()
        self.conv_p4= QtGui.QLineEdit('conv_1')
        VBox2_2.addWidget(conv_lbl_1)
        VBox2_2.addWidget(self.conv_p1)
        VBox2_2.addWidget(conv_lbl_2)
        VBox2_2.addWidget(self.conv_p2)
        VBox2_2.addWidget(conv_lbl_3)
        VBox2_2.addWidget(self.conv_p3)
        VBox2_2.addWidget(conv_lbl_4)
        VBox2_2.addWidget(self.conv_p4)
        VBox2_2.setAlignment(QtCore.Qt.AlignCenter)
        
        VBox2_3 = QtGui.QVBoxLayout()
        maxpool_lbl_1 = QtGui.QLabel('Pool Length :')
        maxpool_lbl_2 = QtGui.QLabel('Stride')
        self.maxpool_p1 = QtGui.QSpinBox()
        self.maxpool_p1.setValue(2)
        self.maxpool_p2 = QtGui.QSpinBox()
        self.maxpool_p2.setValue(0)
        VBox2_3.addWidget(maxpool_lbl_1)
        VBox2_3.addWidget(self.maxpool_p1)
        VBox2_3.addWidget(maxpool_lbl_2)
        VBox2_3.addWidget(self.maxpool_p2)
        VBox2_3.setAlignment(QtCore.Qt.AlignCenter)
        
        VBox2_4 = QtGui.QVBoxLayout()
        flatt_lbl_1 = QtGui.QLabel('Flatten Model Dim')
        VBox2_4.addWidget(flatt_lbl_1)
        VBox2_4.setAlignment(QtCore.Qt.AlignCenter)

        VBox2_5 = QtGui.QVBoxLayout()
        dropout_lbl_1 = QtGui.QLabel('DropOut Percentage (Max=1.0)')
        self.dropout_p1 = QtGui.QDoubleSpinBox()
        self.dropout_p1.setRange(0,1)
        self.dropout_p1.setSingleStep(0.01)
        self.dropout_p1.setValue(0.2)
        VBox2_5.addWidget(dropout_lbl_1)
        VBox2_5.addWidget(self.dropout_p1)
        VBox2_5.setAlignment(QtCore.Qt.AlignCenter)
        
        VBox2_6 = QtGui.QVBoxLayout()
        dense_lbl_1 = QtGui.QLabel('Number of Neurons: ')
        dense_lbl_2 = QtGui.QLabel('Activation')
        dense_lbl_3 = QtGui.QLabel('Using W_Constraint(MaxNorm)')
        self.dense_p1 = QtGui.QSpinBox()
        self.dense_p1.setRange(0,10000)
        self.dense_p1.setValue(1024)
        self.dense_p2 = QtGui.QComboBox()
        self.dense_p3 = QtGui.QSpinBox()
        self.dense_p3.setValue(0)
        VBox2_6.addWidget(dense_lbl_1)
        VBox2_6.addWidget(self.dense_p1)
        VBox2_6.addWidget(dense_lbl_2)
        VBox2_6.addWidget(self.dense_p2)
        VBox2_6.addWidget(dense_lbl_3)
        VBox2_6.addWidget(self.dense_p3)
        VBox2_6.setAlignment(QtCore.Qt.AlignCenter)

        self.zeropad_widget = QtGui.QWidget()
        self.zeropad_widget.setLayout(VBox2_1)
        self.conv_widget = QtGui.QWidget()
        self.conv_widget.setLayout(VBox2_2)
        self.maxpool_widget = QtGui.QWidget()
        self.maxpool_widget.setLayout(VBox2_3)
        self.flatt_widget = QtGui.QWidget()
        self.flatt_widget.setLayout(VBox2_4)
        self.dropout_widget = QtGui.QWidget()
        self.dropout_widget.setLayout(VBox2_5)
        self.dense_widget = QtGui.QWidget()
        self.dense_widget.setLayout(VBox2_6)

        actv_lbl = QtGui.QLabel('Activation')
        self.actv_box = QtGui.QComboBox()
        for i in range(len(activ)):
            if i==8:
                self.actv_box.addItem('None',i)
            else:
                self.actv_box.addItem(activ[i],i)
        
        actv_vb = QtGui.QVBoxLayout()
        actv_vb.addWidget(actv_lbl)
        actv_vb.addWidget(self.actv_box)
        actv_vb.setAlignment(QtCore.Qt.AlignCenter)
        self.actv_widget = QtGui.QWidget()
        self.actv_widget.setLayout(actv_vb)
        

        keras_g3 = QtGui.QGroupBox('Set Layer Parameters')
        keras_g3.setAlignment(QtCore.Qt.AlignCenter)
        VBox2 = QtGui.QGridLayout()
        VBox2.addWidget(self.zeropad_widget)
        VBox2.addWidget(self.conv_widget)
        VBox2.addWidget(self.maxpool_widget)
        VBox2.addWidget(self.flatt_widget)
        VBox2.addWidget(self.dropout_widget)
        VBox2.addWidget(self.dense_widget)
        VBox2.addWidget(self.actv_widget)
        keras_g3.setLayout(VBox2)

        keras_g4 = QtGui.QGroupBox('Add Layer')
        keras_g4.setAlignment(QtCore.Qt.AlignCenter)
        VBox3 = QtGui.QVBoxLayout()
        params_btn = QtGui.QPushButton('Add Layer')
        params_btn.clicked.connect(self.layer_add)
        VBox3.addWidget(params_btn)
        keras_g4.setLayout(VBox3)


        self.layerList = QtGui.QListWidget()
        remove_layer_btn = QtGui.QPushButton('Remove Layer')
        remove_layer_btn.clicked.connect(self.remove_layer)
        clear_layer_btn = QtGui.QPushButton('Clear Layer')
        clear_layer_btn.clicked.connect(self.clear_layer)
        set_lastlayer_btn = QtGui.QPushButton('Set Last Layer')
        set_lastlayer_btn.clicked.connect(self.last_layer)
        layerGrid = QtGui.QGridLayout()
        layerGrid.addWidget(self.layerList,0,0,3,1)
        layerGrid.addWidget(remove_layer_btn,1,1)
        layerGrid.addWidget(clear_layer_btn,2,1)
        layerGrid.addWidget(set_lastlayer_btn,0,1)
        self.layer_widget = QtGui.QWidget()
        self.layer_widget.setLayout(layerGrid)
        layer_widget2 = QtGui.QGridLayout()
        layer_widget2.addWidget(self.layer_widget)
        self.layers = QtGui.QGroupBox('Show Layers')
        self.layers.setCheckable(True)
        self.layers.setChecked(False)
        self.layers.toggled.connect(self.show_layers)
        self.layers.setLayout(layer_widget2)
        self.layers.setFixedSize(535,150)
        self.layer_widget.setVisible(False)

        build_model = QtGui.QPushButton('Save Layers')
        build_model.clicked.connect(self.build_keras_model)
        self.status_model = QtGui.QLabel('')

        for c in self.conv_p3,self.dense_p2:
            c.addItem('ReLu (Rectifier Linear Unit)',0)
            c.addItem('tanh',1)
            c.addItem('Linear',2)
            c.addItem('Sigmoid',3)
            c.addItem('Hard Sigmoid',4)
            c.addItem('SoftSign',5)
            c.addItem('SoftPlus',6)
            c.addItem('SoftMax',7)
            c.addItem('None',8)
        
        grid = QtGui.QGridLayout()
        for w in keras_g1,keras_g2:
            w.setFixedSize(175,96)
        keras_g4.setFixedSize(175,56)
        keras_g3.setFixedSize(185,220)
        grid.addWidget(keras_g1,0,0)
        grid.addWidget(opt_gbox,0,1,1,4)
        grid.addWidget(keras_g2,1,0)
        grid.addWidget(keras_g3,1,1)
        grid.addWidget(keras_g4,1,2)
        grid.addWidget(self.layers,2,0,1,3)
        grid.addWidget(self.status_model,3,0)
        grid.addWidget(build_model,3,1)
        grid.addWidget(close_button,3,2)

        self.zeropad_widget.setVisible(True)
        self.conv_widget.setVisible(False)
        self.maxpool_widget.setVisible(False)
        self.flatt_widget.setVisible(False)
        self.dropout_widget.setVisible(False)
        self.dense_widget.setVisible(False)
        self.actv_widget.setVisible(False)

        self.coll_widget = [self.zeropad_widget,self.conv_widget,
                            self.maxpool_widget,self.flatt_widget,
                            self.dropout_widget,self.dense_widget,
                            self.actv_widget]

        self.setLayout(grid)
        self.setWindowTitle("Keras NN Model Interface")

    def optimizer_comp(self,index):
        for i in range(7):
            if i==index:
                self.optimizer_widget[i].setVisible(True)
            else:
                self.optimizer_widget[i].setVisible(False)

    def optimizer_var(self):
        sgd_state = self.sgd_nest.checkState()
        if sgd_state == 0:
            nest_state = False
        else:
            nest_state = True
        index = self.comp_cbox.currentIndex()
        sgd = [self.sgd_lr.value(),self.sgd_mn.value(),self.sgd_dc.value(),
               nest_state]
        rms = [self.rms_lr.value(),self.rms_rho.value(),self.rms_eps.value(),
               self.rms_dc.value()]
        adag = [self.adag_lr.value(),self.adag_eps.value(),
                self.adag_dc.value()]
        adad = [self.adad_lr.value(),self.adad_rho.value(),
                self.adad_eps.value(),self.adad_dc.value()]
        adam = [self.adam_lr.value(),self.adam_beta1.value(),
                self.adam_beta2.value(),self.adam_eps.value(),
                self.adam_dc.value()]
        adamax = [self.adamax_lr.value(),self.adamax_beta1.value(),
                  self.adamax_beta2.value(),self.adamax_eps.value(),
                  self.adamax_dc.value()]
        nadam = [self.nadam_lr.value(),self.nadam_beta1.value(),
                 self.nadam_beta2.value(),self.nadam_eps.value(),
                 self.nadam_dc.value()]

        c_arg = [sgd,rms,adag,adad,adam,adamax,nadam]
        del sgd,rms,adag,adad,adam,adamax,nadam,sgd_state,nest_state
        
        return [c_arg[index],index]
    
    def set_params(self,index):
        for i in range(7):
            if i==index:
                self.coll_widget[i].setVisible(True)
            else:
                self.coll_widget[i].setVisible(False)

    def input_dim(self):
        k = self.dim.value()
        self.dim_lbl.setText('{}x{} = {}'.format(str(k),str(k),str(k*k)))
        self.inputDim = self.dim.value()*self.dim.value()
        del k

    def show_layers(self,value):
        self.layer_widget.setVisible(value)

    def layer_add(self):
        index = self.CBox1.currentIndex()
        zp_arg = [self.zeropad_p1.value()]
        conv_arg = [self.conv_p1.value(),self.conv_p2.value(),
                    activ[self.conv_p3.currentIndex()],self.conv_p4.text()]
        mp_arg = [self.maxpool_p1.value(),self.maxpool_p2.value()]
        flt_arg = [1]
        do_arg = [self.dropout_p1.value()]
        dns_arg = [self.dense_p1.value(),activ[self.dense_p2.currentIndex()],
                                               self.dense_p3.value()]
        actv_arg = [activ[self.actv_box.currentIndex()]]

        if index==1:
            if conv_arg[3] in self.conv_nama:
                conv_arg[3]=conv_arg[3]+'_'+str(self.count)
        arg = [zp_arg,conv_arg,mp_arg,flt_arg,do_arg,dns_arg,actv_arg]
        self.count += 1
        teks,lay = model_dev(index,arg[index],self.count,self.inputDim)
        if lay is None:
            self.status_model.setText('Maxpool and Dropout layer\n'
                                      'cannot be the first layer')
            self.count -= 1
            return
        x = self.layerList.currentRow()
        if self.count > 2 and x >= 0:
            self.current_layers.insert(x,lay)
            self.layerList.insertItem(x+1,teks[0])
            self.string_layers.append(x,index)
        else:
            self.layerList.addItem(teks[0])
            if index==1:
                self.conv_nama.append(conv_arg[3])
            self.current_layers.append(lay)
            self.string_layers.append(index)
            if index==5:
                self.last_layer_arg = [index,arg[index],self.count,self.inputDim]
                self.last_layer_err = self.count
        self.layerList.setCurrentRow(-1)
        self.ind = index

    def build_keras_model(self):
        cek_flatt=0
        cek_dense = 0
        for k in range(len(self.string_layers)):
            if self.string_layers[k]==3:
                cek_flatt=k
            if self.string_layers[k]==5:
                cek_dense = k
                break

        if cek_dense-cek_flatt> 1:
            self.status_model.setText('Before Dense image should be flatten')
            return
        
        if self.layerList.count()==0:
            self.status_model.setText('No Model')
            return

        cek = self.count-self.last_layer_err

        if cek > 2:
            self.status_model.setText('Last 2 layer must be Dense layer')
            return
        elif cek < 3:
            if self.ind not in [5,6]:
                self.status_model.setText('Last layer should consist Dense'
                                          'or Activation layer')
                return
            if self.last_layer_arg[1][0]!=self.num_classes:
                self.status_model.setText('Number of neurons not match with'
                                          'the given num_classes')
                return

        arg,index = self.optimizer_var()
        optimizer = optimizer_setup(index,arg)
        cek = str(self.layerList.item(0).text())
        cek = cek.split('input')
        if len(cek)< 2:
            self.status_model.setText('First Layer must consist\n'
                                      'inputDim parameter')
            return
        self.status_model.setText('Model layers saved')
        self.keras_model_build = [self.current_layers,optimizer,
                                  self.epo_p1.value(),self.bs_p1.value()]

    def save(self):
        if self.m == 0:
            return self.keras_model_build
        elif self.m == 1:
            return self.sgd_model_build
        elif self.m == 2:
            return self.mlp_model_build

    def remove_layer(self):
        x = self.layerList.currentRow()
        if x==0:
            self.status_model.setText('First Layer must consist\n'
                                      'inputDim parameter')
            return
        self.layerList.takeItem(x)
        del self.current_layers[x]
        del self.string_layers[x]
        self.layerList.setCurrentRow(-1)
        self.count -= 1

    def clear_layer(self):
        self.layerList.clear()
        self.count = 0
        self.current_layers = []
        self.status_model.setText('Layers deleted')

    def last_layer(self):
        cek = self.count-self.last_layer_err
        if cek > 2:
            self.status_model.setText('Last 2 layer must be Dense layer')
            return
        elif cek < 3:
            if self.ind not in [5,6]:
                self.status_model.setText('Last layer should consist Dense'
                                          'or Activation layer')
                return
            
        cek = self.last_layer_err-1
        self.layerList.takeItem(cek)
        del self.current_layers[self.last_layer_err-1]
        arg = self.last_layer_arg
        dns_arg = arg[1]
        dns_arg[0] = self.num_classes
        teks,lay = model_dev(arg[0],dns_arg,arg[2],arg[3])
        self.layerList.insertItem(self.last_layer_err-1,teks[0])
        self.current_layers.insert(self.last_layer_err-1,lay)
            
        del dns_arg,arg,teks,lay
            
        self.layerList.setCurrentRow(-1)

        

if __name__ == '__main__':
    import sys
    m=0
    n=28
    app = QtGui.QApplication(sys.argv)
    dialog = Window(m,n)
    dialog.exec_()
    if dialog.result()==0:
        v1 = dialog.save()

    

    sys.exit()
        

        

