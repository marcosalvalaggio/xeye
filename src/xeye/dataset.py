import cv2 
import os
from sklearn.model_selection import train_test_split
import numpy as np
import time


class Dataset:
    """
    A class for collecting a dataset of images using OpenCV.

    Attributes:
        index (int): The index of the camera to be used for creating the dataset.
        label (list): The list of names of the different types of images to be collected.
        num (int): The number of frames to shoot for every image type.
        height (int): The height of the individual frames.
        width (int): The width of the individual frames.
        standby_time (float): The number of seconds to wait between every frame.
        perc (int): The percentage of data to use for training (the remaining will be used for validation).
        name (str): The name of the dataset.
    
    Examples:
        >>> import xeye
        >>> data = xeye.Dataset()
        >>> data.setup()
        >>> data.var_control()
        >>> data.preview()
        >>> data.rgb() # or data.gray()
        >>> data.compress_train_test()
        >>> data.compress_all()
        >>> data.just.compress()
    """
    index = 0
    label = []
    num = 0
    height = 0
    width = 0
    _class_dict = {}
    _tensor = {}
    standby_time = 0
    _statusGray = 0
    _statusRGB = 0
    perc = 0
    name = "dataset_raw"


    @classmethod
    def setup(cls) -> None:
        """
        Starts the terminal interface for setting up the parameters of the dataset.

        Returns:
            None
        """
        # clear terminal 
        if(os.name == 'posix'): #unix
            os.system('clear')
        else: #windows
            os.system('cls')
        # camera setting control 
        print('--- CAMERA SETTING ---')
        cls.index = int(input('Select the index of the camera that you want to use for creating the dataset: '))
        if cls.index == -1:
            raise ValueError('Insert valid camera index...')
        camera = cv2.VideoCapture(cls.index)
        if camera.isOpened() == False:
            raise ValueError('Insert valid camera index...')
        camera.release()
        # set how many type of images do you want to collect
        cls.label = []
        print('\n')
        print('--- IMAGE SETTINGS ---')
        img_types = int(input('Num. of types of images to scan: '))
        if img_types == 0: 
            raise ValueError('Number of images types must be greather than 0')
        for i in range(0,img_types):
            l = str(input(f"Name of image type ({i+1}): "))
            cls.label.append(l)
        # folder building
        for lab in cls.label:
            if not os.path.exists(lab):
                os.mkdir(lab)
        # number of frames 
        cls.num = int(input('Num. of frames to shoot for every image type: '))
        if cls.num == 0 or cls.num < 0:
            raise ValueError('You cant inizialize frames number equal or below zero...')
        # Image shaping phase
        cls.height = int(input('Single frame HEIGHT: '))
        if cls.height == 0 or cls.height < 0:
            raise ValueError('Frame HEIGHT must be greather than 0')
        cls.width = int(input('Single frame WIDTH: '))
        if cls.width == 0 or cls.width < 0:
            raise ValueError('Frame WIDTH must be greather than 0')

        # Waiting time in shooting loop
        cls.standby_time = float(input('num. of waiting time (in sec.) between every frame: '))
        if cls.standby_time < 0:
            raise ValueError('waiting time must be grater than 0...')


    @classmethod
    def preview(cls) -> None:
        """
        Opens the camera stream on a window for checking the framing of the image.

        Returns: 
            None
        """
        print('\n')
        print('--- PREVIEW ---')
        camera = cv2.VideoCapture(cls.index)
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


    @classmethod
    def gray(cls) -> None:
        """
        Method for shooting images in grayscale.

        Returns:
            None 
        """
        print('\n')
        print('--- START TAKING PHOTOS ---')
        camera = cv2.VideoCapture(cls.index)
        # Index for files name 
        i = 0
        for folder in cls.label:
            count = 0
            print(f'Press [b] on the keyboard to start data collection of image type: [{folder}]')
            userinput = input()
            if userinput != 'b':
                print("Wrong Input...press 'b'")
            while count < cls.num:
                status, frame = camera.read()
                if not status:
                    print("frame doesn't been captured")
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.startWindowThread()
                cv2.imshow(f"Camera View for image type [{folder}]", gray)
                gray = cv2.resize(gray, (cls.width, cls.height))
                cv2.imwrite(folder+'/'+ str(cls.label[i]) + str(count) + '.png', gray)
                count=count+1
                time.sleep(cls.standby_time)
                if cv2.waitKey(1) == ord('q'):
                    break
            i += 1
        camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        # set status
        cls._statusGray = 1
        cls._statusRGB = 0


    @classmethod
    def rgb(cls) -> None:
        """
        Method for shooting images in RGB.

        Returns:
            None
        """
        print('\n')
        print('--- START TAKING PHOTOS ---')
        camera = cv2.VideoCapture(cls.index)
        # Index for files name 
        i = 0
        for folder in cls.label:
            count = 0
            print(f'Press [b] on the keyboard to start data collection of image type: [{folder}]')
            userinput = input()
            if userinput != 'b':
                print("Wrong Input...press 'b'")
                exit()
            while count < cls.num:
                status, frame = camera.read()
                if not status:
                    print("frame doesn't been captured")
                    break
                cv2.startWindowThread()
                cv2.imshow(f"Camera View for image type [{folder}]", frame)
                frame = cv2.resize(frame, (cls.width, cls.height))
                cv2.imwrite(folder+'/'+ str(cls.label[i]) + str(count) + '.png', frame)
                count=count+1
                time.sleep(cls.standby_time)
                if cv2.waitKey(1) == ord('q'):
                    break
            i += 1
        camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        # Set status 
        cls._statusGray = 0
        cls._statusRGB = 1
    

    @classmethod
    def compress_train_test(cls) -> None:
        """
        Saves the images shot in datasets divided by train and test like the mnist dataset.
    
        Raises:
            ValueError: If both rgb and gray functions have not been called before compressing a dataset.
            ValueError: If the percentage value for images in the test set is less than or equal to 0.
    
        Returns:
            None
        """
        # data control
        if cls._statusRGB == 0 and cls._statusGray == 0:
            raise ValueError('You have to call rgb or gray function before compress a dataset...')
        print('\n')
        print('--- DATASET SETTING ---')
        cls.perc = float(input('percentage of images in the test set (0,1): '))
        if cls.perc <= 0:
            raise ValueError('percentage value must be greater than 0...')
        # index for image type 
        i = 0
        # X
        if cls._statusGray == 1:
            cls._tensor['X'] = np.empty((0,cls.height,cls.width))
        else:
            cls._tensor['X'] = np.empty((0,cls.height,cls.width,3))
        # y 
        cls._tensor['y'] = np.empty((0))
        # (append) loop 
        for lab in cls.label:
            j = 0
            if cls._statusGray == 1:
                cls._class_dict['t'+str(i)] = np.empty((cls.num, cls.height, cls.width))
            else:
                cls._class_dict['t'+str(i)] = np.empty((cls.num, cls.height, cls.width, 3))
            # loop for convert image format  
            for file in os.listdir(lab):
                if cls._statusGray == 1:
                    img = cv2.imread(lab + '/' + file, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(lab + '/' + file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # save
                cls._class_dict['t'+str(i)][j] = img
                j += 1
            # unique final _tensors 
            cls._tensor['X'] = np.append(cls._tensor['X'], cls._class_dict['t'+str(i)], axis = 0)
            cls._tensor['y'] = np.append(cls._tensor['y'], np.repeat(i, cls.num, axis = 0))
            i += 1
        # create dataset (mnist style)
        cls._tensor['X'] = cls._tensor['X'].astype('uint8')
        cls._tensor['y'] = cls._tensor['y'].astype('uint8')
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls._tensor['X'], cls._tensor['y'], test_size=cls.perc, random_state=123)
        np.savez('dataset.npz', X_train=cls.X_train, X_test=cls.X_test, y_train=cls.y_train, y_test=cls.y_test)


    @classmethod
    def compress_all(cls) -> None:
        """
        Saves the images shot in a unique dataset.
    
        Raises:
            ValueError: If both rgb and gray functions have not been called before compressing a dataset.
        
        Returns:
            None
        """
        # data control
        if cls._statusRGB == 0 and cls._statusGray == 0:
            raise ValueError('You have to call rgb or gray function before compress a dataset...')
        # index for image type 
        i = 0
        # X
        if cls._statusGray == 1:
            cls._tensor['X'] = np.empty((0,cls.height,cls.width))
        else:
            cls._tensor['X'] = np.empty((0,cls.height,cls.width,3))
        # y 
        cls._tensor['y'] = np.empty((0))
        # (append) loop 
        for lab in cls.label:
            j = 0
            if cls._statusGray == 1:
                cls._class_dict['t'+str(i)] = np.empty((cls.num, cls.height, cls.width))
            else:
                cls._class_dict['t'+str(i)] = np.empty((cls.num, cls.height, cls.width, 3))
            # loop for convert image format
            for file in os.listdir(lab):
                if cls._statusGray == 1:
                    img = cv2.imread(lab + '/' + file, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(lab + '/' + file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # save 
                cls._class_dict['t'+str(i)][j] = img
                j += 1
            # unique final _tensors 
            cls._tensor['X'] = np.append(cls._tensor['X'], cls._class_dict['t'+str(i)], axis = 0)
            cls._tensor['y'] = np.append(cls._tensor['y'], np.repeat(i, cls.num, axis = 0))
            i += 1
        # create dataset (mnist style)
        cls._tensor['X'] = cls._tensor['X'].astype('uint8')
        cls._tensor['y'] = cls._tensor['y'].astype('uint8')
        np.savez('datasetall.npz', x = cls._tensor['X'], y = cls._tensor['y'])


    @classmethod
    def just_compress(cls) -> None:
        """
        Saves the images shot in a unique dataset without saving the y variable containing the type of the single image.
    
        Raises:
            ValueError: If both rgb and gray functions have not been called before compressing a dataset.
        
        Returns:
            None
        """
        # data control
        if cls._statusRGB == 0 and cls._statusGray == 0:
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
            cls.name = name 
        # index for image type 
        i = 0
        # X
        if cls._statusGray == 1:
            cls._tensor['X'] = np.empty((0,cls.height,cls.width))
        else:
            cls._tensor['X'] = np.empty((0,cls.height,cls.width,3))
        # (append) loop 
        for lab in cls.label:
            j = 0
            if cls._statusGray == 1:
                cls._class_dict['t'+str(i)] = np.empty((cls.num, cls.height, cls.width))
            else:
                cls._class_dict['t'+str(i)] = np.empty((cls.num, cls.height, cls.width, 3))
            # loop for convert image format
            for file in os.listdir(lab):
                if cls._statusGray == 1:
                    img = cv2.imread(lab + '/' + file, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(lab + '/' + file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # save
                cls._class_dict['t'+str(i)][j] = img
                j += 1
            # unique final _tensors
            cls._tensor['X'] = np.append(cls._tensor['X'], cls._class_dict['t'+str(i)], axis = 0)
            i += 1
        # create dataset (mnist style)
        cls._tensor['X'] = cls._tensor['X'].astype('uint8')
        np.savez(f'{cls.name}.npz', x = cls._tensor['X'])


    @classmethod
    def var_control(cls) -> None:
        """
        Print the parameters specified in the setup method about the dataset to create.
        """
        print('\n')
        print('--- PARAMETERS CONTROL ---')
        print(f'Camera index: {cls.index}')
        print(f'Labels of images types: {cls.label}')
        print(f'Num. of images for types: {cls.num}')
        print(f'Single frame HEIGHT: {cls.height}')
        print(f'Single frame WIDTH: {cls.width}')
        print(f'Waiting time between frames: {cls.standby_time}')
        print(f'Percentage of images in train dataset: {cls.perc}')
        print(f'StatusGray: {cls._statusGray}')
        print(f'StatusRGB: {cls._statusRGB}')
