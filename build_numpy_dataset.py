import numpy as np
import scipy.io

sample_length = 512 #1024

samples_to_take = 200 #100

data_folder = "./data"

test_split = 0.2

file_names = [["Ball 0.007", 118], ["Ball 0.014", 185], ["Ball 0.021", 222], ["Inner race 0.007", 105], ["Inner race 0.014", 169], ["Inner race 0.021", 209], ["Outer Race Orthogonal", 144], ["Outer Race Centered", 130], ["Outer Race Opposite", 156], ["Normal", 97]]
n_files = len(file_names)
n_classes = n_files

x_train = []
y_train = []

samples_to_take_test = int(np.floor(samples_to_take * test_split))
samples_to_take_train = samples_to_take - samples_to_take_test

x_test = []
y_test = []

for idx, file_name in enumerate(file_names) :
    f = scipy.io.loadmat(f"{data_folder}/{file_name[1]}.mat")
    data = f[f"X{file_name[1]:03}_FE_time"]
    n_samples = int(data.shape[0] / sample_length)
    data = np.reshape(data[0:n_samples * sample_length], (n_samples, sample_length))[:samples_to_take]
    ids = np.random.permutation(data.shape[0])
    data = data[ids]
    x_train.append(data[:samples_to_take_train])
    x_test.append(data[samples_to_take_train:])

    labels = np.zeros((samples_to_take, n_classes))
    labels[:, idx] = 1.
    y_train.append(labels[:samples_to_take_train])
    y_test.append(labels[samples_to_take_train:])

x_train = np.reshape(np.array(x_train), (samples_to_take_train * n_files, 1, sample_length, 1))
x_test = np.reshape(np.array(x_test), (samples_to_take_test * n_files, 1, sample_length, 1))
y_train = np.reshape(np.array(y_train), (samples_to_take_train * n_files, n_classes))
y_test = np.reshape(np.array(y_test), (samples_to_take_test * n_files, n_classes))

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)

np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)

print("training split")
print(f"data shape: {x_train.shape}")
print(f"labels shape: {y_train.shape}")

print("testing split")
print(f"data shape: {x_test.shape}")
print(f"labels shape: {y_test.shape}")
