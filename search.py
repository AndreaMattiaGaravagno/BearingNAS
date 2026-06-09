from BearingNAS import BearingNAS
from pathlib import Path
import numpy as np

#load_dataset
#input data should be in the shape (batch_size, 1, n_samples, n_channels)
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")

x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

val_split=0.1

#whether or not to cache datasets in memory
#if the dataset cannot fit in the main memory, the application will crash
cache = True

#target: STM32F030F4P6
#106 CoreMark, 16 kiB Flash, 4 kiB RAM
#MACC_upper_bound = 1060000 #CoreMark * 10^4
#flash_upper_bound = 16384
#ram_upper_bound = 4096

#target: STM32C011F6P6
#114 CoreMark, 32 kiB Flash, 6 kiB RAM
#MACC_upper_bound = 114000 #CoreMark * 10^4
#flash_upper_bound = 32768
#ram_upper_bound = 6144

#target:  LSM6DSO16IS
#n.a. CoreMark, 32 kiB Flash, 8 kiB RAM
MACC_upper_bound = 1140000 #last CoreMark * 10^4
flash_upper_bound = 32768
ram_upper_bound = 8192

time_budget = 60 * 60 * 60 #s - seconds

BearingNAS = BearingNAS(ram_upper_bound, flash_upper_bound, MACC_upper_bound, time_budget, x_train, y_train, val_split=val_split, cache=cache)

architecture_exists = BearingNAS.search()
if architecture_exists :
    BearingNAS.train_resulting_architecture(16, 0.01, 100)
    test_acc = BearingNAS.test_tflite_model(x=x_test, y=y_test)
    print(f"Test Accuracy: {test_acc}")
