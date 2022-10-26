import cv2 
import os
from sklearn.model_selection import train_test_split
import numpy as np
import time


class dataset:

    # inizialized variable
    index = 0
    label = []
    num = 0
    height = 0
    width = 0
    class_dict = {}
    tensor = {}
    standby_time = 0
    statusGray = 0
    statusRGB = 0
    perc = 0
    name = "dataset_raw"

    # --------------------
    # init image parameters
    # --------------------
    def init(self):

        # clear terminal 
        if(os.name == 'posix'): #unix
            os.system('clear')
        else: #windows
            os.system('cls')
        
        
        # camera setting control 
        print('--- CAMERA SETTING ---')
        self.index = int(input('Select the index of the camera that you want to use for creating the dataset: '))
        if self.index == -1:
            raise ValueError('Insert valid camera index...')
        camera = cv2.VideoCapture(self.index)
        if camera.isOpened() == False:
            raise ValueError('Insert valid camera index...')
        camera.release()

        # set how many type of images do you want to collect
        self.label = []
        print('\n')
        print('--- IMAGE SETTINGS ---')
        img_types = int(input('Num. of types of images to scan: '))
        if img_types == 0: # informarsi sul raise error 
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


    # --------------------
    # preview 
    # --------------------    
    def preview(self):

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
    def gray(self):
        
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
        self.statusGray = 1
        self.statusRGB = 0

    
    # -----------
    # rgb imges
    # -----------
    def rgb(self):

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
    def compressTrainTest(self):
        # data control
        if self.statusRGB == 0 and self.statusGray == 0:
            raise ValueError('You have to call rgb or gray function before compress a dataset...')
        print('\n')
        print('--- DATASET SETTING ---')
        self.perc = float(input('percentage of images in the test set (0,1): '))
        if self.perc <= 0:
            raise ValueError('percentage value must be greater than 0...')
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.tensor['X'], self.tensor['y'], test_size=self.perc, random_state=123)
        np.savez('dataset.npz', X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test)


    # -----------
    # compressAll
    # -----------
    def compressAll(self):
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
    def justCompress(self):
        # data control
        if self.statusRGB == 0 and self.statusGray == 0:
            raise ValueError('You have to call rgb or gray function before compress a dataset...')
        # insert name for the compress file 
        print('\n')
        print('--- DATASET SETTING ---')
        name = input('Select a name for the compressed file: ')
        # check the name for the dataset 
        if len(name) == 0:
            raise ValueError("Insert a valide name for the compress file...")
        if name == "0":
            pass
        else:
            self.name = name 
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
        np.savez(f'{self.name}.npz', x = self.tensor['X'])



    # ------------------
    # control function 
    # ------------------

    def varControl(self):
        print('\n')
        print('--- PARAMETERS CONTROL ---')
        print(f'camera index: {self.index}')
        print(f'labels of images types: {self.label}')
        print(f'num. of images for types: {self.num}')
        print(f'single frame HEIGHT: {self.height}')
        print(f'single frame WIDTH: {self.width}')
        print(f'waiting time between frames: {self.standby_time}')
        print(f'percentage of images in train dataset: {self.perc}')
        print(f'statusGray: {self.statusGray}')
        print(f'statusRGB: {self.statusRGB}')



class fastDataset:
    
    def __init__(self, index, img_types, label, num, height, width, stand_by_time):
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

    def init(self):

        # clear terminal 
        if(os.name == 'posix'): #unix
            os.system('clear')
        else: #windows
            os.system('cls')
        
        # camera setting
        if self.index == -1:
            raise ValueError('Insert valid camera index...')
        camera = cv2.VideoCapture(self.index)
        if camera.isOpened() == False:
            raise ValueError('Insert valid camera index...')

        # set how many type of images do you want to collect
        if self.img_types == 0: # informarsi sul raise error 
            raise ValueError('Number of images types must be greather than 0')
        # folder building
        if self.label == []:
            raise ValueError('Not valid names for images types...')
        if len(self.label) != self.img_types:
            raise ValueError("You must have a number of labels equal to the number of images types selected...")
        for lab in self.label:
            if not os.path.exists(lab):
                os.mkdir(lab)

        # number of frames 
        if self.num == 0 or self.num < 0:
            raise ValueError('You cant inizialize frames number equal or below zero...')
        

        # Image shaping phase
        if self.height == 0 or self.height < 0:
            raise ValueError('Frame HEIGHT must be greather than 0')
        if self.width == 0 or self.width < 0:
            raise ValueError('Frame WIDTH must be greather than 0')

        # Waiting time in shooting loop
        if self.standby_time < 0:
            raise ValueError('waiting time must be grater than 0...')




    # --------------------
    # preview 
    # --------------------      
    def preview(self):

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
    def gray(self):
        
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
    def rgb(self):

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
    def compressTrainTest(self, perc = 0.1):
        # percentage control 
        if perc <= 0:
            raise ValueError('percentage value must be greater than 0...')
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
    def compressAll(self):
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
    def justCompress(self, name = "dataset_raw"):
        # data control
        if self.statusRGB == 0 and self.statusGray == 0:
            raise ValueError('You have to call rgb or gray function before compress a dataset...')
        # check the name for the dataset 
        if len(str(name)) == 0:
            raise ValueError("Insert a valide name for the compressed file...")
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



    def varControl(self):
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



class buildDataset:

    def __init__(self, path, label, color = True, split = True, perc = 0.1):
        self.path = path
        self.label = label 
        self.color = color 
        self.split = split
        self.perc = perc
        self.height = 0
        self.width = 0 
        self.tensor = {}

    def sizer(self):
        _height = []
        _width = []
        for path in self.path:
            data = np.load(path)
            x = data['x']
            _height.append(x.shape[1])
            _width.append(x.shape[2])

        if len(set(_height)) == 1:
            self.height = _height[0]
        else:
            raise ValueError("Datasets doesn't have the same dimensions...")
        
        if len(set(_width)) == 1:
            self.width = _width[0]
        else:
            raise ValueError("Datasets doesn't have the same dimensions...")

    def build(self):
        # call the crawler function for check the dimensions of the images
        self.sizer()
        # control if the path and label lists have the same length
        if len(self.path) != len(self.label):
            raise ValueError("Path and label lists doesn't have the same length...")
        # Create the tensor X
        if self.color == True:
            self.tensor['X'] = np.empty((0,self.height,self.width,3))
        else: 
            self.tensor['X'] = np.empty((0,self.height,self.width))
        # array for label y 
        self.tensor['y'] = np.empty((0))
        # loop 
        for i in range(len(self.path)):
            data = np.load(f'{self.path[i]}')
            x = data['x']
            self.tensor['X'] = np.append(self.tensor['X'], x, axis = 0)
            self.tensor['y'] = np.append(self.tensor['y'], np.repeat(self.label[i], x.shape[0], axis = 0))
        # set data type for memory management 
        self.tensor['X'] = self.tensor['X'].astype('uint8')
        self.tensor['y'] = self.tensor['y'].astype('uint8')
        # save dataset 
        if self.split == True:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.tensor['X'], self.tensor['y'], test_size=self.perc, random_state=123)
            np.savez('dataset.npz', X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test)
        else: 
            np.savez('datasetall.npz', x = self.tensor['X'], y = self.tensor['y'])




if __name__ == '__main__':

    
    data = dataset()
    data.init()
    data.preview()
    data.gray()
    data.rgb()
    data.compressTrainTest()
    data.compressAll()
    data.justCompress()
    data.varControl()

    '''
    ### test with fastDataset 
    index = 0
    img_types = 1
    label = ['test']
    num = 20
    height = 100
    width = 100
    standby_time = 0
    perc = 0.2

    # class call 
    data = fastDataset(index = index, img_types = img_types, label = label, num = num, height = height, width = width, stand_by_time = standby_time)
    data.init()
    data.preview()
    data.varControl()
    data.gray()
    data.rgb()
    data.compressAll()
    data.compressTrainTest(perc = perc)
    '''
