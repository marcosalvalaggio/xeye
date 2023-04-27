import xeye

# define parameters values
index = 0
img_types = 2
label = ['keyboard', 'mouse']
num = 20
height = 100
width = 100
standby_time = 0
# percentage of images in the test set 
perc = 0.2

data = xeye.FastDataset(index = index, img_types = img_types, label = label,
                        num = num, height = height, width = width, stand_by_time = standby_time)
data.preview()
data.rgb() # or data.gray()
data.compress_train_test(perc = perc)
data.compress_all()
data.just_compress("batch_test")