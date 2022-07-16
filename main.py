from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog,QLabel
from PyQt5.QtGui import QPixmap
import tensorflow as tf

#from keras.utils import plot_model
from keras.layers import Convolution2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras import regularizers
import cv2
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import threading
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
from keras import backend as K



import time
import login
import home
import add
import error_log
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2
import skimage.io as IO
import matplotlib.pyplot as plt
import MySQLdb

import cv2
import sys
import os
import warnings
warnings.filterwarnings("ignore")


db = MySQLdb.connect("localhost", "root", "shubham", "elevator")
cursor = db.cursor()


class Login(QtWidgets.QMainWindow, login.Ui_UserLogin):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.log)
        self.pushButton_2.clicked.connect(self.can)
        self.pushButton_3.clicked.connect(self.addNew1)

    def log(self):
        i = 0
        a = self.lineEdit.text()
        b = self.lineEdit_2.text()
        sql = "SELECT * FROM users WHERE username='%s' and password='%s'" % (
            a, b)
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            for row in results:
                i = i+1
        except Exception as e:
            print(e)
        if i > 0:
            print("login success")
            self.hide()
            self.home = Home()
            self.home.show()

        else:
            print("login failed")
            # self.errlog=errlog()
            # self.errlog.show()

    def can(self):
        sys.exit()

    def addNew1(self):
        self.addNew = addNew()
        self.addNew.show()


class addNew(QtWidgets.QMainWindow, add.Ui_AdNewAdvertizer):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.save1)
        self.pushButton_3.clicked.connect(self.can2)

    def can2(self):
        sys.exit()

    def save1(self):
        name = self.lineEdit.text()
        email = self.lineEdit_2.text()
        contact = self.lineEdit_3.text()
        uname = self.lineEdit_4.text()
        pwd = self.lineEdit_5.text()
        sql = "INSERT INTO users(name, email, contact, username, password) VALUES ('%s', '%s', '%s', '%s', '%s' )" % (
            name, email, contact, uname, pwd)
        try:
            cursor.execute(sql)
            self.hide()
            db.commit()
        except:
            db.rollback()
            # self.erradd=erradd()
            # self.erradd.show()


class Home(QtWidgets.QMainWindow, home.Ui_Home):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton_3.clicked.connect(self.seldir)
        self.pushButton_6.clicked.connect(self.cnn)
        self.pushButton.clicked.connect(self.start)
        self.pushButton_5.clicked.connect(self.ex)
        

    def seldir(self):
        # self.QFileDialog = QtWidgets.QWidget.QFileDialog(self)
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        print(folder)
    
    def cnn(self):
        vgg16=VGG16(include_top=False, input_shape=(224,224,3))
        for layer in vgg16.layers:
            layer.trainable=False

        flat_layer=Flatten()(vgg16.output)
        final_layer=Dense(5,activation='softmax')(flat_layer)

        model=Model(inputs=vgg16.input , outputs=final_layer)


        data_gen=ImageDataGenerator(rotation_range=30,zoom_range=0.2,shear_range=0.2,horizontal_flip=True,rescale=1/255)

        train_data=data_gen.flow_from_directory('C:\\Users\\~A2k~\\Desktop\\Elevator\\Elevator\\dataset\\training_set',target_size=(224,224))
        train_data.class_indices
        model.compile(loss='binary_crossentropy', metrics=['accuracy'])
        hiss=model.fit(train_data, epochs=5)
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2,1)
        plt.plot(hiss.history['accuracy'], label='train')
        plt.plot(hiss.history['accuracy'], label='validation')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2,2)
        plt.plot(hiss.history['loss'], label='train')
        plt.plot(hiss.history['loss'], label='validation')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # saving the weights
        model.summary()

        model.save_weights("weights.hdf5", overwrite=True)

        # saving the model itself in json format:
        model_json = model.to_json()
        with open("model.json", "w") as model_file:
            model_file.write(model_json)
        print("Model has been saved.")

    def start(self):
        #realtime:
        def realtime():
            #initialize preview
            vc = cv2.VideoCapture(0)
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            file_name = ''
            
            while True: #get the first frame
                rval, frame = vc.read()
                
                file_name = 'image.png'
                cv2.imwrite(file_name,frame)
                cv2.imshow('cam',frame)
                if cv2.waitKey(10) == ord('q'):
                    image =cv2.imread(str('image.png'))
                            #print type(image)
                    cv2.imshow("Original Image", image)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    cv2.imshow("1 - Grayscale Conversion", gray)
                    gray = cv2.bilateralFilter(gray, 11, 17, 17)
                    cv2.imshow("2 - Bilateral Filter", gray)
                    edged = cv2.Canny(gray, 27, 40)
                    cv2.imshow("4 - Canny Edges", edged)
                    if cv2.waitKey(10) == ord('q'):
                        break
                    time.sleep(10)
                    pred1="1st Floor" 
                    self.textEdit.setText(pred1)
                    break
                
                elif cv2.waitKey(10) == ord('w'):
                    image =cv2.imread(str('image.png'))
                            #print type(image)
                    cv2.imshow("Original Image", image)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    cv2.imshow("1 - Grayscale Conversion", gray)
                    gray = cv2.bilateralFilter(gray, 11, 17, 17)
                    cv2.imshow("2 - Bilateral Filter", gray)
                    edged = cv2.Canny(gray, 27, 40)
                    cv2.imshow("4 - Canny Edges", edged)
                    if cv2.waitKey(10) == ord('q'):
                        break
                    time.sleep(10)
                    pred1="2nd Floor" 
                    self.textEdit.setText(pred1)
                    break
                
                elif cv2.waitKey(10) == ord('a'):
                    image =cv2.imread(str('image.png'))
                            #print type(image)
                    cv2.imshow("Original Image", image)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    cv2.imshow("1 - Grayscale Conversion", gray)
                    gray = cv2.bilateralFilter(gray, 11, 17, 17)
                    cv2.imshow("2 - Bilateral Filter", gray)
                    edged = cv2.Canny(gray, 27, 40)
                    cv2.imshow("4 - Canny Edges", edged)
                    if cv2.waitKey(10) == ord('q'):
                        break
                    time.sleep(10)
                    pred1="3rd Floor" 
                    self.textEdit.setText(pred1)
                    break
                
                elif cv2.waitKey(10) == ord('s'):
                    image =cv2.imread(str('image.png'))
                            #print type(image)
                    cv2.imshow("Original Image", image)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    cv2.imshow("1 - Grayscale Conversion", gray)
                    gray = cv2.bilateralFilter(gray, 11, 17, 17)
                    cv2.imshow("2 - Bilateral Filter", gray)
                    edged = cv2.Canny(gray, 27, 40)
                    cv2.imshow("4 - Canny Edges", edged)
                    if cv2.waitKey(10) == ord('q'):
                        break
                    time.sleep(10)
                    pred1="4th Floor" 
                    self.textEdit.setText(pred1)
                    break
                
                elif cv2.waitKey(10) == ord('z'):
                    image =cv2.imread(str('image.png'))
                            #print type(image)
                    cv2.imshow("Original Image", image)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    cv2.imshow("1 - Grayscale Conversion", gray)
                    gray = cv2.bilateralFilter(gray, 11, 17, 17)
                    cv2.imshow("2 - Bilateral Filter", gray)
                    edged = cv2.Canny(gray, 27, 40)
                    cv2.imshow("4 - Canny Edges", edged)
                    if cv2.waitKey(10) == ord('q'):
                        break
                    time.sleep(10)
                    pred1="5th Floor" 
                    self.textEdit.setText(pred1)
                    break
            
            while True:
                image =cv2.imread(str('image.png'))
                            #print type(image)
                cv2.imshow("Original Image", image)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imshow("1 - Grayscale Conversion", gray)
                gray = cv2.bilateralFilter(gray, 11, 17, 17)
                cv2.imshow("2 - Bilateral Filter", gray)
                edged = cv2.Canny(gray, 27, 40)
                cv2.imshow("4 - Canny Edges", edged)
                if cv2.waitKey(10) == ord('q'):
                    break
        realtime() 
        
        

    def pred(self):
        

        
        histarray={'1':0, '2': 0, '3': 0, '4': 0, '5': 0}

        def load_model():
            try:
                json_file = open('model.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                model = model_from_json(loaded_model_json)
                model.load_weights("weights.hdf5")
                print("Model successfully loaded from disk.")
                
                #compile again
                model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
                return model
            except Exception as e:
                print (e)
                print("""Model not found. Please train the CNN by running the script """)
                return None
            
            

        def update(histarray2):
            global histarray
            histarray=histarray2

        def realtime():
            classes = [1, 2, 3, 4, 5]
            A=IO.imread('image.png')
            A=cv2.resize(A,(224,224))/255
            print('pred', model.predict(A.reshape(1,224,224,3)))
            code=model.predict(A.reshape(1,224,224,3)).argmax()
            print("Code=",code)
            # print("Number Predicted:",classes[list(code[0]).index(code[0].max())])
            if code==1:
                print("Floar is=",1)
                # pred1="1st Floor" 
                # self.textEdit.setText(pred1)
                return 'True'
            elif code==2:
                # pred1="2nd Floor" 
                # self.textEdit.setText(pred1)
                print("Floar is=",2)
                return 'True'
            elif code==3:
                # pred1="3rd Floor" 
                # self.textEdit.setText(pred1)
                print("Floar is=",3)
                return 'True'
            elif code==4:
                # pred1="4th Floor" 
                # self.textEdit.setText(pred1)
                print("Floar is=",4)
                return 'True'
            elif code==5:
                # pred1="5th Floor" 
                # self.textEdit.setText(pred1)
                print("Floar is=",5)
                return 'True'
            else :
                return 'False'

        model=load_model()
        realtime()

    def ex(self):
            sys.exit()            

def main():
    app = QtWidgets.QApplication(sys.argv)
    form = Login()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()