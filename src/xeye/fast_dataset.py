import cv2 
import os
from sklearn.model_selection import train_test_split
import numpy as np
import time


class FastDataset:
    
    def __init__(self, index: int, img_types: int, label: list[str], num: int, height: int, width: int, stand_by_time: float) -> None:
    # inizialized variable
        self.index = index
        self.img_types = img_types
        self.label = label
        self.num = num
        self.height = height
        self.width = width
        self.class_dict = {}
        self.tensor = {}
        self.standby_time = stand_by_time
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
        # Waiting time in shooting loop
        if self.standby_time < 0:
            raise ValueError('(standby_time) waiting time must be grater than 0...')


    # --------------------
    # preview 
    # --------------------      
    def preview(self)->None:
        """
        Open the camera stream on a window. Helpful for checking the framing of the image.
        """
        print('\n')
        print('--- PREVIEW ---')
        camera = cv2.VideoCapture(self.index)
        while(True):
            status, frame = camera.read()
            if not status:
                print("frame doesn't been captured")
                break
            font = cv2.FONT_HERSHEY_COMPLEX
            text = 'click on the image window and then press [q] on the keyboard to quit preview'
            cv2.putText(frame,text,(5,50),font,0.8,(124,252,0),2)  #text,coordinate,font,size of text,color,thickness of font
            cv2.startWindowThread()
            cv2.imshow("Camera PreView", frame)
            if cv2.waitKey(1) == ord('q'):
                print('preview closed')
                camera.release()
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                break


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
                cv2.imshow(f"Camera View for image type [{folder}]",gray)
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
        self.statusGray = 1
        self.statusRGB = 0


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
                cv2.imshow(f"Camera View for image type [{folder}]", frame)
                frame = cv2.resize(frame, (self.width, self.height))
                cv2.imwrite(folder+'/'+ str(self.label[i]) + str(count) + '.png', frame)
                count=count+1
                time.sleep(self.standby_time)
                if cv2.waitKey(1) == ord('q'):
                    break
            i += 1
        camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        # Set status 
        self.statusGray = 0
        self.statusRGB = 1


    # -----------
    # compressTrainTest
    # -----------
    def compress_train_test(self, perc: float = 0.1)->None:
        """
        Save the images shot in datasets divided by train and test like the mnist dataset.
        """
        # percentage control 
        if perc <= 0:
            raise ValueError('(perc) Percentage value must be greater than 0...')
        # data control
        if self.statusRGB == 0 and self.statusGray == 0:
            raise ValueError('You have to call rgb or gray function before compress a dataset...')
        # index for image type 
        i = 0
        # X
        if self.statusGray == 1:
            self.tensor['X'] = np.empty((0,self.height,self.width))
        else:
            self.tensor['X'] = np.empty((0,self.height,self.width,3))
        # y 
        self.tensor['y'] = np.empty((0))
        # (append) loop 
        for lab in self.label:
            j = 0
            if self.statusGray == 1:
                self.class_dict['t'+str(i)] = np.empty((self.num, self.height, self.width))
            else:
                self.class_dict['t'+str(i)] = np.empty((self.num, self.height, self.width, 3))
            # loop for convert image format  
            for file in os.listdir(lab):
                if self.statusGray == 1:
                    img = cv2.imread(lab + '/' + file, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(lab + '/' + file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # save in tensor class 
                self.class_dict['t'+str(i)][j] = img
                j += 1
            # unique final tensors 
            self.tensor['X'] = np.append(self.tensor['X'], self.class_dict['t'+str(i)], axis = 0)
            self.tensor['y'] = np.append(self.tensor['y'], np.repeat(i, self.num, axis = 0))
            i += 1
        # create dataset (mnist style)
        self.tensor['X'] = self.tensor['X'].astype('uint8')
        self.tensor['y'] = self.tensor['y'].astype('uint8')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.tensor['X'], self.tensor['y'], test_size=perc, random_state=123)
        np.savez('dataset.npz', X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test)


    # -----------
    # compressAll
    # -----------
    def compress_all(self)->None:
        """
        Save the images shot in a unique dataset.
        """
        # data control
        if self.statusRGB == 0 and self.statusGray == 0:
            raise ValueError('You have to call rgb or gray function before compress a dataset...')
        # index for image type 
        i = 0
        # X
        if self.statusGray == 1:
            self.tensor['X'] = np.empty((0,self.height,self.width))
        else:
            self.tensor['X'] = np.empty((0,self.height,self.width,3))
        # y 
        self.tensor['y'] = np.empty((0))
        # (append) loop 
        for lab in self.label:
            j = 0
            if self.statusGray == 1:
                self.class_dict['t'+str(i)] = np.empty((self.num, self.height, self.width))
            else:
                self.class_dict['t'+str(i)] = np.empty((self.num, self.height, self.width, 3))
            # loop for convert image format
            for file in os.listdir(lab):
                if self.statusGray == 1:
                    img = cv2.imread(lab + '/' + file, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(lab + '/' + file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # save in tensor class 
                self.class_dict['t'+str(i)][j] = img
                j += 1
            # unique final tensors 
            self.tensor['X'] = np.append(self.tensor['X'], self.class_dict['t'+str(i)], axis = 0)
            self.tensor['y'] = np.append(self.tensor['y'], np.repeat(i, self.num, axis = 0))
            i += 1
        # create dataset (mnist style)
        self.tensor['X'] = self.tensor['X'].astype('uint8')
        self.tensor['y'] = self.tensor['y'].astype('uint8')
        np.savez('datasetall.npz', x = self.tensor['X'], y = self.tensor['y'])



    # -----------
    # justCompress
    # -----------
    def just_compress(self, name: str = "dataset_raw")->None:
        """
        Save the images shot in a unique dataset without saving the y variable containing the type of the single image.
        """
        # data control
        if self.statusRGB == 0 and self.statusGray == 0:
            raise ValueError('You have to call rgb or gray function before compress a dataset...')
        # check the name for the dataset 
        if len(str(name)) == 0:
            raise ValueError("name: Insert a valide name for the compressed file...")
        # index for image type 
        i = 0
        # X
        if self.statusGray == 1:
            self.tensor['X'] = np.empty((0,self.height,self.width))
        else:
            self.tensor['X'] = np.empty((0,self.height,self.width,3))
        # (append) loop 
        for lab in self.label:
            j = 0
            if self.statusGray == 1:
                self.class_dict['t'+str(i)] = np.empty((self.num, self.height, self.width))
            else:
                self.class_dict['t'+str(i)] = np.empty((self.num, self.height, self.width, 3))
            # loop for convert image format
            for file in os.listdir(lab):
                if self.statusGray == 1:
                    img = cv2.imread(lab + '/' + file, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(lab + '/' + file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # save in tensor class 
                self.class_dict['t'+str(i)][j] = img
                j += 1
            # unique final tensors
            self.tensor['X'] = np.append(self.tensor['X'], self.class_dict['t'+str(i)], axis = 0)
            i += 1
        # create dataset (mnist style)
        self.tensor['X'] = self.tensor['X'].astype('uint8')
        np.savez(f'{name}.npz', x = self.tensor['X'])



    def var_control(self)->None:
        """
        Print the parameters specified in the init method about the dataset to create.
        """
        print('\n')
        print('--- PARAMETERS CONTROL ---')
        print(f'camera index: {self.index}')
        print(f'num. of images types: {self.img_types}')
        print(f'labels of images types: {self.label}')
        print(f'num. of images for types: {self.num}')
        print(f'single frame HEIGHT: {self.height}')
        print(f'single frame WIDTH: {self.width}')
        print(f'waiting time between frames: {self.standby_time}')
        print(f'percentage of images in train dataset: {self.perc}')
        print(f'statusGray: {self.statusGray}')
        print(f'statusRGB: {self.statusRGB}')
