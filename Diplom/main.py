from cam import *
from PyQt5 import QtCore, QtGui, QtWidgets
import tensorflow,skimage,cv2
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
import string
import sys
import time

model = tensorflow.keras.models.load_model("my_modelBalanced.h5")
Alphabet=[x for x in list(string.ascii_uppercase)]
Alphabet.append("DEL")
Alphabet.append("NOTHING")
Alphabet.append("SPACE")
def num_to_letter(num):
    letter = Alphabet[num]
    return letter

def Test_image(path_to_image):

    img_file =path_to_image
    imageSize = 64
    img_file = cv2.resize(img_file, (64,64))
    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
    img_arr = np.asarray(img_file).reshape((-1, imageSize, imageSize,3))
    return img_arr

def predict(img_arr):
    Pletter = num_to_letter(np.argmax(model.predict(img_arr)))
    predict = "это буква - " + Pletter
    return predict

class MyWin(QtWidgets.QMainWindow):
    
    def __init__(self,parent=None):
        QtWidgets.QWidget.__init__(self,parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
       


        self.ui.pushButton.clicked.connect(self.MyFunction)
    def setText(self,prediction):
        self.ui.textEdit.setText("")
        self.ui.textEdit.setText(prediction)
    def settime(self):
        timer = self.ui.textEdit_2.toPlainText()
        try:
            timer = int(timer)
        except:
            timer =1
        return timer

    def MyFunction(self):
        cap = cv2.VideoCapture(0)
        counter =0
        start_time = time.perf_counter()
        while(True):    
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("WebCam Output", gray)
            img = Test_image(gray)
            prediction = predict(img)
            current_time = time.perf_counter()
            if -start_time+current_time >= self.settime() :
                print(prediction)
                
                self.setText(prediction)
                start_time = time.perf_counter()
            #counter = counter +1           
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyWin()
    myapp.show()
    sys.exit(app.exec_())




    
