import cv2 
import os
from sklearn.model_selection import train_test_split
import numpy as np
import time
from .fast_dataset import FastDataset


class ManualDataset(FastDataset):

    def __init__(self, index: int, img_types: int, label: list[str], num: int, height: int, width: int)->None:
    # inizialized variable
        self.index = index
        self.img_types = img_types
        self.label = label
        self.num = num
        self.height = height
        self.width = width
        self.class_dict = {}
        self.tensor = {}
        self.statusGray = 0
        self.statusRGB = 0
        # clear terminal 
        if(os.name == 'posix'): #unix
            os.system('clear')
        else: #windows
            os.system('cls')
        # camera setting
        if self.index == -1:
            raise ValueError('(index) Insert valid camera index...')
        camera = cv2.VideoCapture(self.index)
        if camera.isOpened() == False:
            raise ValueError('(index) Insert valid camera index...')
        # set how many type of images do you want to collect
        if self.img_types == 0: # informarsi sul raise error 
            raise ValueError('(img_types) Number of images types must be greather than 0')
        if type(self.img_types) != int:
            raise TypeError('(img_types) passed a string, not an integer')
        # folder building
        if self.label == []:
            raise ValueError('(label) Not valid names for images types...')
        if len(self.label) != self.img_types:
            raise ValueError("(label) You must have a number of labels equal to the number of images types selected...")
        for lab in self.label:
            if not os.path.exists(lab):
                os.mkdir(lab)
        # number of frames 
        if self.num == 0 or self.num < 0:
            raise ValueError('(num) You cant inizialize frames number equal or below zero...')
        # Image shaping phase
        if self.height == 0 or self.height < 0:
            raise ValueError('(height) Frame HEIGHT must be greather than 0')
        if self.width == 0 or self.width < 0:
            raise ValueError('(width) Frame WIDTH must be greather than 0')


    # -----------
    # rgb imges
    # -----------
    def rgb(self)->None:
        """
        Method for shooting images in RGB. 
        """
        print('\n')
        print('--- START TAKING PHOTOS ---')
        camera = cv2.VideoCapture(self.index)
        # Index for files name 
        i = 0
        for folder in self.label:
            count = 0
            print(f'Press [b] on the keyboard to start data collection of image type: [{folder}]')
            userinput = input()
            if userinput != 'b':
                print("Wrong Input...press 'b'")
                exit()
            while count < self.num:
                status, frame = camera.read()
                if not status:
                    print("frame doesn't been captured")
                    break
                cv2.startWindowThread()
                cv2.imshow(f"Camera View for image type [{folder}], Press [s] on the keyboard to save the image nr: {count}", frame)
                frame = cv2.resize(frame, (self.width, self.height))
                if cv2.waitKey(1) == ord('s'):
                    cv2.imwrite(folder+'/'+ str(self.label[i]) + str(count) + '.png', frame)
                    count=count+1
                    cv2.destroyAllWindows()
                else:
                    pass
                if cv2.waitKey(1) == ord('q'):
                    break           
            i += 1
        camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        # Set status 
        self.statusGray = 0
        self.statusRGB = 1


    # ---------------
    # grayscale images 
    #----------------
    def gray(self)->None:
        """
        Method for shooting images in grayscale. 
        """
        print('\n')
        print('--- START TAKING PHOTOS ---')
        camera = cv2.VideoCapture(self.index)
        # Index for files name 
        i = 0
        for folder in self.label:
            count = 0
            print(f'Press [b] on the keyboard to start data collection of image type: [{folder}]')
            userinput = input()
            if userinput != 'b':
                print("Wrong Input...press 'b'")
            while count < self.num:
                status, frame = camera.read()
                if not status:
                    print("frame doesn't been captured")
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.startWindowThread()
                cv2.imshow(f"Camera View for image type [{folder}], Press [s] on the keyboard to save the image nr: {count}",gray)
                gray = cv2.resize(gray, (self.width, self.height))
                if cv2.waitKey(1) == ord('s'):
                    cv2.imwrite(folder+'/'+ str(self.label[i]) + str(count) + '.png', gray)
                    count=count+1
                    cv2.destroyAllWindows()
                else:
                    pass
                if cv2.waitKey(1) == ord('q'):
                    break
            i += 1
        camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        # set status
        self.statusGray = 1
        self.statusRGB = 0
