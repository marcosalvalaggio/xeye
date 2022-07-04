import cv2 
import os
from sklearn.model_selection import train_test_split
import numpy as np
from pyfiglet import Figlet
import time


class dataset():

    # inizialized variable
    index = 0
    label = []
    num = 0
    height = 0
    width = 0
    class_dict = {}
    tensor = {}
    stand_by_time = 0
    statusGray = 0
    statusRGB = 0
    perc = 0

    # --------------------
    # init image parameters
    # --------------------
    def Init(self):

        # clear terminal 
        if(os.name == 'posix'): #unix
            os.system('clear')
        else: #windows
            os.system('cls')
        
        # Welcome lines:
        title = Figlet(font='slant')
        print(title.renderText('DataPipe'))
        print('Just answer the questions and than start making DL model based on your own dataset.')
        print('\n')

        # camera setting control 
        print('--- CAMERA SETTING ---')
        self.index = int(input('Select index of the camera that you want to use for crate the dataset: '))
        if self.index == -1:
            raise TypeError('Insert valid camera index...')
        camera = cv2.VideoCapture(self.index)
        if camera.isOpened() == False:
            raise TypeError('Insert valid camera index...')

        # set how many type of images do you want to collect
        self.label = []
        print('\n')
        print('--- IMAGE SETTING ---')
        n = int(input('How many types of images do you want to scan: '))
        if n == 0: # informarsi sul raise error 
            raise TypeError('Number of images types must be greather than 0')
        for i in range(0,n):
            l = str(input(f"Name of image type ({i+1}): "))
            self.label.append(l)
        # folder building
        for lab in self.label:
            if not os.path.exists(lab):
                os.mkdir(lab)

        # number of frames 
        self.num = int(input('How many frames do you want to shoot for every image type: '))
        if self.num == 0 or self.num < 0:
            raise TypeError('You cant inizialize frames number equal or below zero...')
        

        # Image shaping phase
        self.height = int(input('Single frame HEIGHT: '))
        if self.height == 0 or self.height < 0:
            raise TypeError('Frame HEIGHT must be greather than 0')
        self.width = int(input('Single frame WIDTH: '))
        if self.width == 0 or self.width < 0:
            raise TypeError('Frame WIDTH must be greather than 0')

        # Waiting time in shooting loop
        self.stand_by_time = float(input('num. of waiting time (in sec.) between every frame: '))
        if self.stand_by_time < 0:
            raise TypeError('waiting time must be grater than 0...')

    
    # ---------------
    # grayscale images 
    #----------------
    def Gray(self):
        
        print('\n')
        print('--- START TAKING PHOTOS ---')

        camera = cv2.VideoCapture(self.index)

        # Index for files name 
        i = 0
        for folder in self.label:
          
            count = 0
            print("Press 'b' on keyboard to start data collection for image type "+folder)
            userinput = input()
            if userinput != 'b':
                print("Wrong Input...press 'b'")

            while count < self.num:
                
                status, frame = camera.read()

                if not status:
                    print("frame doesn't been captured")
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                cv2.imshow("Camera View",gray)

                gray = cv2.resize(gray, (self.width, self.height))

                cv2.imwrite(folder+'/'+ str(self.label[i]) + str(count) + '.png', gray)
                

                count=count+1

                time.sleep(self.stand_by_time)

                if cv2.waitKey(1) == ord('q'):
                    break

            i += 1

        camera.release()
        cv2.destroyAllWindows()
        # set status
        self.statusGray = 1
        self.statusRGB = 0

    
    # -----------
    # rgb imges
    # -----------
    def Rgb(self):

        camera = cv2.VideoCapture(self.index)

        # Index for files name 
        i = 0
        for folder in self.label:

            count = 0

            print("Press 'b' on keyboard to start data collection for image type "+folder)
            userinput = input()
            if userinput != 'b':
                print("Wrong Input...press 'b'")
                exit()

            while count < self.num:

                status, frame = camera.read()

                if not status:
                    print("frame doesn't been captured")
                    break

                cv2.imshow("Camera View", frame)

                frame = cv2.resize(frame, (self.width, self.height))

                cv2.imwrite(folder+'/'+ str(self.label[i]) + str(count) + '.png', frame)

                count=count+1

                time.sleep(self.stand_by_time)

                if cv2.waitKey(1) == ord('q'):
                    break
            
            i += 1

        camera.release()
        cv2.destroyAllWindows()
        # Set status 
        self.statusGray = 0
        self.statusRGB = 1
    


    def CompressTrainTest(self):
        print('\n')
        print('--- DATASET SETTING ---')
        self.perc = float(input('percentage of images in train dataset: '))
        if self.perc <= 0:
            raise TypeError('percentage value must be greater than 0...')
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
                    img = cv2.imread(lab + '/' + file, cv2.COLOR_BGR2RGB)
                # save in tensor class 
                self.class_dict['t'+str(i)][j] = img
                j += 1
            # unique final tensors 
            self.tensor['X'] = np.append(self.tensor['X'], self.class_dict['t'+str(i)], axis = 0)
            self.tensor['y'] = np.append(self.tensor['y'], np.repeat(i+1, self.num, axis = 0))
            i += 1
        # create dataset (mnist style)
        self.tensor['X'] = self.tensor['X'].astype('uint8')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.tensor['X'], self.tensor['y'], test_size=self.perc, random_state=123)
        np.savez('dataset.npz', X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test)


    def CompressAll(self):
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
                    img = cv2.imread(lab + '/' + file, cv2.COLOR_BGR2RGB)
                # save in tensor class 
                self.class_dict['t'+str(i)][j] = img
                j += 1
            # unique final tensors 
            self.tensor['X'] = np.append(self.tensor['X'], self.class_dict['t'+str(i)], axis = 0)
            self.tensor['y'] = np.append(self.tensor['y'], np.repeat(i+1, self.num, axis = 0))
            i += 1
        # create dataset (mnist style)
        self.tensor['X'] = self.tensor['X'].astype('uint8')
        np.savez('datasetall.npz', x = self.tensor['X'], y = self.tensor['y'])



    # ------------------
    # control function 
    # ------------------
    def VarControl(self):
        print(self.index)
        print(self.label)
        print(self.num)
        print(self.height)
        print(self.width)
        print(self.statusGray)
        print(self.statusRGB)



if __name__ == '__main__':

    data = dataset()
    data.Init()
    data.Gray()
    #data.Rgb()
    data.CompressTrainTest()
    #data.CompressAll()
    data.VarControl()
