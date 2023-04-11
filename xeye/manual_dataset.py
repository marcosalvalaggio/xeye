import cv2 
import os
from .fast_dataset import FastDataset
from typing import List


class ManualDataset(FastDataset):

    def __init__(self, index: int, img_types: int, label: List[str], num: int, height: int, width: int, stand_by_time: float = 0) -> None:
        super().__init__(index, img_types, label, num, height, width, stand_by_time)


    def gray(self) -> None:
        """
        Method for manually shooting images in grayscale.

        Returns:
            None 
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
        self._statusGray = 1
        self._statusRGB = 0


    def rgb(self) -> None:
        """
        Method for manually shooting images in RGB.

        Returns:
            None
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
        self._statusGray = 0
        self._statusRGB = 1
