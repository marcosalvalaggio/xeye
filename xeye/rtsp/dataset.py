import cv2 
import os
from sklearn.model_selection import train_test_split
import numpy as np
import time


class Dataset:
    """
    A class for collecting a dataset of images in a rtsp stream using OpenCV.
    """

    def __init__(self):
        # clear terminal 
        if(os.name == 'posix'): #unix
            os.system('clear')
        else: #windows
            os.system('cls')
        # camera setting control 
        print('--- RTSP SETTING ---')
        self.rtsp = str(input('insert the rtsp: '))
        # set how many type of images do you want to collect
        self.label = []
        print('\n')
        print('--- IMAGE SETTINGS ---')
        img_types = int(input('Num. of types of images to scan: '))
        if img_types == 0: 
            raise ValueError('Number of images types must be greather than 0')
        for i in range(0,img_types):
            l = str(input(f"Name of image type ({i+1}): "))
            self.label.append(l)
        # folder building
        for lab in self.label:
            if not os.path.exists(lab):
                os.mkdir(lab)
        # number of frames 
        self.num = int(input('Num. of frames to shoot for every image type: '))
        if self.num == 0 or self.num < 0:
            raise ValueError('You cant inizialize frames number equal or below zero...')
        # Image shaping phase
        self.height = int(input('Single frame HEIGHT: '))
        if self.height == 0 or self.height < 0:
            raise ValueError('Frame HEIGHT must be greather than 0')
        self.width = int(input('Single frame WIDTH: '))
        if self.width == 0 or self.width < 0:
            raise ValueError('Frame WIDTH must be greather than 0')
        # Waiting time in shooting loop
        self.standby_time = float(input('num. of waiting time (in sec.) between every frame: '))
        if self.standby_time < 0:
            raise ValueError('waiting time must be grater than 0...')
        self.perc=0.25
        self._class_dict = {}
        self._tensor = {}


    def preview(self):
        """
        Opens the camera stream on a window for checking the framing of the image.

        Returns: 
            None
        """
        print('\n')
        print('--- PREVIEW ---')
        camera = cv2.VideoCapture(self.rtsp)
        if not camera.isOpened():
            raise ValueError('Unable to open camera stream.')
        
        while True:
            status, frame = camera.read()
            if not status:
                raise ValueError('Unable to read camera stream.')
            frame = cv2.resize(frame, (1080, 720))
            font = cv2.FONT_HERSHEY_COMPLEX
            text = 'click on the image window and then press [q] on the keyboard to quit preview'
            cv2.putText(frame,text,(5,50),font,0.6,(124,252,0),2)  #text,coordinate,font,size of text,color,thickness of font
            cv2.startWindowThread()
            cv2.imshow("RTSP camera Preview, click q to quit preview", frame)
            if cv2.waitKey(1) == ord('q'):
                print('preview closed')
                camera.release()
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                break


    def gray(self) -> None:
        """
        Method for shooting images in grayscale.

        Returns:
            None 
        """
        print('\n')
        print('--- START TAKING PHOTOS ---')
        camera = cv2.VideoCapture(self.rtsp)
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
                cv2.imshow(f"Camera View for image type [{folder}]", gray)
                gray = cv2.resize(gray, (self.width, self.height))
                cv2.imwrite(folder+'/'+ str(self.label[i]) + str(count) + '.png', gray)
                count=count+1
                time.sleep(self.standby_time)
                if cv2.waitKey(1) == ord('q'):
                    break
            i += 1
        camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        # set status
        self._statusGray = 1
        self._statusRGB = 0