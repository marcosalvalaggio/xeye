import xeye

# define parameters values
source = 0
img_types = 2
label = ['keyboard', 'mouse']
num = 20
height = 100
width = 100
data = xeye.ManualDataset(source = source, img_types = img_types, label = label, num = num, height = height, width = width)
data.preview()
data.rgb() # or data.gray()
data.compress_train_test(perc=0.2)
data.compress_all()
data.just_compress(name="batch_test")