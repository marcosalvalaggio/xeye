from ..usb.dataset import Dataset
import os
from typing import List


class RTSPDataset(Dataset):
    """
    A class for shooting and saving images in grayscale or RGB using OpenCV.

    Attributes:
        rtsp (str): rtsp connection string.
        img_types (int): the number of types of images to collect.
        label (List[str]): a list of strings that represent the name of the directories where the images will be saved.
        num (int): the number of frames to capture for each image type.
        height (int): the height of the frames to capture.
        width (int): the width of the frames to capture.
        standby_time (float): the time to wait before capturing each frame. 

    Examples: 
        >>> import xeye.rtsp as rtsp
        >>> # define parameters values
        >>> rtsp = "rtsp://admin:hik12345@192.168.10.6/ISAPI/Streaming/channels/101"
        >>> img_types = 2
        >>> label = ['keyboard', 'mouse']
        >>> num = 20
        >>> height = 100
        >>> width = 100
        >>> standby_time = 0
        >>> data = rtsp.Dataset(rtsp=rtsp, img_types=img_types, label=label, num=num, height=height, width=width, stand_by_time=standby_time)
        >>> data.preview()
        >>> data.rgb() # or data.gray()
        >>> data.compress_train_test(perc=0.2)
        >>> data.compress_all()
        >>> data.just_compress(name="batch_test")
    """
    def __init__(self, rtsp: str, img_types: int, label: List[str], num: int, height: int, width: int, stand_by_time: float):
        self.rtsp = rtsp
        self.img_types = img_types
        self.label = label
        self.num = num
        self.height = height
        self.width = width
        self._class_dict = {}
        self._tensor = {}
        self.standby_time = stand_by_time
        self._statusGray = 0
        self._statusRGB = 0
        # clear terminal 
        if(os.name == 'posix'): # unix
            os.system('clear')
        else: # windows
            os.system('cls')
        # move index to rtsp
        self.index = self.rtsp
        # img num
        if self.img_types == 0: 
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