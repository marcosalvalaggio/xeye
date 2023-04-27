import xeye

data = xeye.Dataset()
data.setup()
data.preview()
data.rgb() # or data.gray()
data.compress_train_test()
data.compress_all()
data.just_compress()
