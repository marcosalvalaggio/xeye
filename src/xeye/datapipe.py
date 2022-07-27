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
        self.index = int(input('Select index of the camera that you want to use for create the dataset: '))
        if self.index == -1:
            raise TypeError('Insert valid camera index...')
        camera = cv2.VideoCapture(self.index)
        if camera.isOpened() == False:
            raise TypeError('Insert valid camera index...')

        # set how many type of images do you want to collect
        self.label = []
        print('\n')
        print('--- IMAGE SETTINGS ---')
        img_types = int(input('How many types of images do you want to scan: '))
        if img_types == 0: # informarsi sul raise error 
            raise TypeError('Number of images types must be greather than 0')
        for i in range(0,img_types):
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
        self.standby_time = float(input('num. of waiting time (in sec.) between every frame: '))
        if self.standby_time < 0:
            raise TypeError('waiting time must be grater than 0...')


    # --------------------
    # preview 
    # --------------------    
    def preview(self):

        camera = cv2.VideoCapture(self.index)

        while(True):

            status, frame = camera.read()

            if not status:
                print("frame doesn't been captured")
                break
            
            font = cv2.FONT_HERSHEY_COMPLEX
            text = 'click on image window and then press [q] on keyboard to quit preview'
            cv2.putText(frame,text,(0,50),font,0.8,(124,252,0),2)  #text,coordinate,font,size of text,color,thickness of font

            cv2.imshow("Camera PreView", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()



    
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

                time.sleep(self.standby_time)

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
    def rgb(self):

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

                time.sleep(self.standby_time)

                if cv2.waitKey(1) == ord('q'):
                    break
            
            i += 1

        camera.release()
        cv2.destroyAllWindows()
        # Set status 
        self.statusGray = 0
        self.statusRGB = 1
    


    def compressTrainTest(self):
        print('\n')
        print('--- DATASET SETTING ---')
        self.perc = float(input('percentage of images in test dataset: '))
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
                    img = cv2.imread(lab + '/' + file)
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


    def compressAll(self):
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



class dataset2:
    
    def __init__(self, index, img_types, label, num, height, width, stand_by_time, perc):
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
        self.perc = 1

    def init(self):

        # clear terminal 
        if(os.name == 'posix'): #unix
            os.system('clear')
        else: #windows
            os.system('cls')
        
        # camera setting
        if self.index == -1:
            raise TypeError('Insert valid camera index...')
        camera = cv2.VideoCapture(self.index)
        if camera.isOpened() == False:
            raise TypeError('Insert valid camera index...')

        # set how many type of images do you want to collect
        if self.img_types == 0: # informarsi sul raise error 
            raise TypeError('Number of images types must be greather than 0')
        # folder building
        if self.label == []:
            raise TypeError('Not valid names for images types...')
        if len(self.label) != self.img_types:
            raise TypeError("You must have a number of labels equal to the number of images types selected...")
        for lab in self.label:
            if not os.path.exists(lab):
                os.mkdir(lab)

        # number of frames 
        if self.num == 0 or self.num < 0:
            raise TypeError('You cant inizialize frames number equal or below zero...')
        

        # Image shaping phase
        if self.height == 0 or self.height < 0:
            raise TypeError('Frame HEIGHT must be greather than 0')
        if self.width == 0 or self.width < 0:
            raise TypeError('Frame WIDTH must be greather than 0')

        # Waiting time in shooting loop
        if self.standby_time < 0:
            raise TypeError('waiting time must be grater than 0...')

        if self.perc <= 0:
            raise TypeError('percentage value must be greater than 0...')



    # --------------------
    # preview 
    # --------------------    
    def preview(self):

        camera = cv2.VideoCapture(self.index)

        while(True):

            status, frame = camera.read()

            if not status:
                print("frame doesn't been captured")
                break
            
            font = cv2.FONT_HERSHEY_COMPLEX
            text = 'click on image window and then press [q] on keyboard to quit preview'
            cv2.putText(frame,text,(0,50),font,0.8,(124,252,0),2)  #text,coordinate,font,size of text,color,thickness of font

            cv2.imshow("Camera PreView", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()



    # ---------------
    # grayscale images 
    #----------------
    def gray(self):
        
        camera = cv2.VideoCapture(self.index)

        # Index for files name 
        i = 0
        for folder in self.label:
          
            count = 0

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

                time.sleep(self.standby_time)

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
    def rgb(self):

        camera = cv2.VideoCapture(self.index)

        # Index for files name 
        i = 0
        for folder in self.label:

            count = 0

            while count < self.num:

                status, frame = camera.read()

                if not status:
                    print("frame doesn't been captured")
                    break

                cv2.imshow("Camera View", frame)

                frame = cv2.resize(frame, (self.width, self.height))

                cv2.imwrite(folder+'/'+ str(self.label[i]) + str(count) + '.png', frame)

                count=count+1

                time.sleep(self.standby_time)

                if cv2.waitKey(1) == ord('q'):
                    break
            
            i += 1

        camera.release()
        cv2.destroyAllWindows()
        # Set status 
        self.statusGray = 0
        self.statusRGB = 1


    def compressTrainTest(self):
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


    def compressAll(self):
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



if __name__ == '__main__':

    
    data = dataset()
    data.init()
    data.preview()
    data.gray()
    #data.rgb()
    data.compressTrainTest()
    #data.compressAll()
    data.varControl()

    '''
    ### test with dataset 2 
    index = 0
    img_types = 1
    label = ['test']
    num = 20
    height = 100
    width = 100
    standby_time = 0
    perc = 0.2

    # class call 
    data = dataset2(index = index, img_types = img_types, label = label, num = num, height = height, width = width, stand_by_time = standby_time, perc = perc)
    data.init()
    data.preview()
    data.varControl()
    data.gray()
    data.rgb()
    data.compressAll()
    data.compressTrainTest()
    '''
