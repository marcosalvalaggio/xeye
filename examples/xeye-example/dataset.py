from xeye import Dataset

# with laptop webcam
# data = Dataset(source=0, img_types=2, label=['a', 'b'], num=10, height=100, width=100, stand_by_time=0)

# IP Hikvision camera
data = Dataset(source='rtsp://admin:password@ip:port/ISAPI/Streaming/channels/101', 
               img_types=2, label=['a', 'b'], num=10, height=100, width=100, stand_by_time=0)
data.preview()
data.gray()
data.compress_train_test(perc=0.2)
data.compress_all()
data.just_compress(name="batch_test")