from PyQt4 import QtGui,QtCore
import sys,os
sys.stderr = sys.stdout
import play_ui
import training_ui


class Window(QtGui.QDialog):
    def __init__(self):
        super(Window, self).__init__()
        
        self.mainUI()
        self.setWindowTitle("FAce Recognition rev1.0 Interface")
        self.setWindowIcon(QtGui.QIcon(os.getcwd()+'/Oxygen.ico'))


    def mainUI(self):
        close_btn = QtGui.QPushButton('Close')
        image_btn = QtGui.QPushButton('Predict an image')
        webcam_btn = QtGui.QPushButton('Webcam Live')
        training_btn = QtGui.QPushButton('Training Interface')

        text ='''
Dilarang keras untuk menyebarluaskan program ini tanpa ijin author.\n
Face Recognition rev1.0 merupakan software untuk mengidentifikasi\n
dan memverifikasi citra wajah yang diperoleh yang kemudian akan dicocokkan\n
dengan database yang ada.\n
\n
Software ini bekerja menggunakan metode neural network sebagai fungsi learning machine.\n
Neural network classifier yang digunakan adalah Keras dan SKlearn.\n
Kedua software tersebut merupakan free source module yang berbasis python programming\n
Best regard thanks to @fchollet sebagai author dari Keras Python.\n
\n
Software ini penuh dibuat dengan menggunakan python.\n
\n
Jika ada pertanyaan atau kendala-kendala mengenai pemakaian software ini\n
silahkan hubungi author.\n
\n
Author : Muhammad Syafiq\n
email : syafiq.photonics.eng@gmail.com\n
       	syafiqguitarist@gmail.com\n'''

        label = QtGui.QLabel(text)
##        upload_btn = QtGui.QPushButton('Upload Data')
        self.status = QtGui.QLabel('')

        close_btn.clicked.connect(self.close)
        image_btn.clicked.connect(self.open_play)
        webcam_btn.clicked.connect(self.cam)
        training_btn.clicked.connect(self.train)
##        upload_btn.clicked.connect(self.uploading)

        Hbox1 = QtGui.QHBoxLayout()
        Hbox1.addWidget(image_btn)
        Hbox1.addWidget(webcam_btn)
        Hbox1.addWidget(training_btn)

        widget = QtGui.QWidget()
        widget.setLayout(Hbox1)

        mainLayout = QtGui.QGridLayout()
        mainLayout.addWidget(label,0,0,1,3)
        mainLayout.addWidget(widget,1,1)
        mainLayout.addWidget(close_btn,2,2)
        mainLayout.addWidget(self.status,2,0)

        self.setLayout(mainLayout)

    def open_play(self):
        win = play_ui.Window(0)
        win.exec_()
        
    def cam(self):
        win = play_ui.Window(1)
        win.exec_()
        
    def train(self):
        win = training_ui.Window()
        win.exec_()        


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    dialog = Window()
    dialog.show()

    sys.exit(app.exec_())
    
