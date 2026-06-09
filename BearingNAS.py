from tensorflow_model_optimization.python.core.keras.compat import keras
from multiprocessing import Process, Queue, Event
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import numpy as np
import subprocess
import datetime
import json
import time
import os
import re

class BearingNAS :
    def __init__(self, max_ram, max_flash, max_macc, time_budget, x=None, y=None, val_split=0.3, cache=False) :
        self.start_time = datetime.datetime.now()

        self.search_learning_rate = 0.001
        self.search_batch_size = 16
        self.epochs_to_evaluate = 100

        self.max_ram = max_ram
        self.max_flash = max_flash
        self.max_macc = max_macc
        self.time_budget = time_budget
        self.x = x
        self.y = y
        self.val_split = val_split
        self.cache = cache
        self.input_shape = x.shape[1:]

        self.num_classes = y.shape[1]

    # k number of kernels of the first convolutional layer
    # c number of cells added upon the first convolutional layer
    # pre-processing pipeline not included in MACC computation
    def model(self, k, c) :
        kernel_size = (1,3)
        pool_size = (1,2)
        pool_strides = (1,2)

        number_of_cells_limited = False
        macc = 0

        if k < 1 or c < 0 :
            return False, None, None, None

        inputs = tf.keras.Input(shape=self.input_shape)

        #convolutional base
        n = k

        #first convolutional layer
        c_in = self.input_shape[2]
        x = tf.keras.layers.Conv2D(n, kernel_size, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        macc = macc + (c_in * kernel_size[0] * kernel_size[1] * x.shape[1] * x.shape[2] * x.shape[3])

        #adding cells
        for i in range (c) :
            if x.shape[2] <= 1 :
                number_of_cells_limited = True
                break

            n = int(n + np.ceil(n / 2))
            x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding='valid')(x)
            c_in = x.shape[3]
            x = tf.keras.layers.Conv2D(n, kernel_size, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            macc = macc + (c_in * kernel_size[0] * kernel_size[1] * x.shape[1] * x.shape[2] * x.shape[3])

        #classifier
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(self.num_classes)(x)
        macc = macc + (x.shape[1] * outputs.shape[1])

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        n_of_params = model.count_params()

        feasible = not number_of_cells_limited

        return feasible, model, macc, n_of_params
    
    def load_time_series(self, batch_size, val_split=False, cache=False) :
        x = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        x = x.shuffle(1000).batch(batch_size)

        if val_split :
            train_ds, val_ds = tf.keras.utils.split_dataset(x, right_size=self.val_split)
            if self.cache :
                train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
                val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            else :
                train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
                val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            return train_ds, val_ds
        else :
            train_ds = x
            if self.cache :
                train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            else :
                train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            return train_ds

    def load_training_set(self, batch_size=1, cache=False) :
        return self.load_time_series(batch_size, cache=cache)

    def load_training_and_validation_sets(self, batch_size=1, cache=False) :
        return self.load_time_series(batch_size, True, cache)

    def compile_model(self, model, learning_rate=0.01) :
         opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

         model.compile(optimizer=opt,
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    def quantize_model(self, model, path_to_tflite_model, train_ds=None) :
        if train_ds == None :
            x = np.ones(self.input_shape)
            x = np.expand_dims(x, axis=0)
            y = np.ones((1, self.num_classes))
            train_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(1)
            def representative_dataset():
                for data in train_ds.take(1) :
                    yield [tf.dtypes.cast(data[0], tf.float32)]
        else :
            def representative_dataset():
                for data in train_ds.rebatch(1).take(150) :
                    yield [tf.dtypes.cast(data[0], tf.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
        tflite_quant_model = converter.convert()

        with open(path_to_tflite_model, 'wb') as f:
            f.write(tflite_quant_model)

    def evaluate_flash_and_peak_ram_occupancy(self, model) :
        #it must be done after one epoch of training, at least
        path_to_tflite_model = 'temp.tflite'
        #quantize model to evaluate its peak RAM occupancy and its Flash occupancy
        self.quantize_model(model, path_to_tflite_model)

        #evaluate its peak RAM occupancy and its Flash occupancy using STMicroelectronics' script named "stm32tflm"
        #found inside the linux package of X-CUBE-AI at the following path:
        #"path/to/en.x-cube-ai-linux_v8.0.1/stm32ai-linux-8.0.1/linux/stm32tflm".
        #The package can be downloaded at https://www.st.com/en/embedded-software/x-cube-ai.html#get-software.
        proc = subprocess.Popen(["./stm32tflm", path_to_tflite_model], stdout=subprocess.PIPE)
        try:
            outs, errs = proc.communicate(timeout=15)
            flash, ram = re.findall(r'\d+', str(outs))
            os.remove(path_to_tflite_model)
        except subprocess.TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()
            os.remove(path_to_tflite_model)
            print("stm32tflm error")
            exit()

        return int(flash), int(ram)

    def evaluate_params_of_architecture_process(self, q, k, c) :
        #mock dataset for evaluating flash and ram occupancy
        #no need to use GPU for such a small training
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        x = np.ones((self.search_batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        y = np.ones((self.search_batch_size, self.num_classes))
        train_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(self.search_batch_size)
        feasible, model, macc, params = self.model(k, c)
        if feasible :
            self.compile_model(model)
            model.fit(train_ds)
            flash, ram = self.evaluate_flash_and_peak_ram_occupancy(model)
            feasibility = macc <= self.max_macc and flash <= self.max_flash and ram <= self.max_ram 
            print("\n\n\n")
            print({'k': k,
                'c': c,
                'RAM': ram if ram <= self.max_ram else f"{ram} (Outside the upper bound of {ram - self.max_ram} Byte)",
                'Flash': flash if flash <= self.max_flash else f"{flash} (Outside the upper bound of {flash - self.max_flash} Byte)",
                'MACC': macc if macc <= self.max_macc else f"{macc} (Outside the upper bound of {macc - self.max_macc} MAC)",
                'params' : params})
            print("\n\n\n")
            q.put(params if True == feasibility else 0)
        else :
            q.put(0)

    def evaluate_params_of_architecture(self, k, c) :
        q = Queue()
        p = Process(target=self.evaluate_params_of_architecture_process, args=(q, k, c,))
        p.start()
        p.join()
        if q.empty() :
            #the machine was not able to train the architecture for one epoch
            params = 0
        else:
            params = q.get()
        return params

    #to be used only after having ferified the feasibility of k,c with the previous function "evaluate_params_of_architecture"
    def evaluate_architecture_process(self, q, k, c) :
        train_ds, validation_ds = self.load_training_and_validation_sets(self.search_batch_size, cache=self.cache)
        model = self.model(k, c)[1]
        self.compile_model(model, self.search_learning_rate)
        min_val_loss = np.finfo(np.float64).max
        for i in range(self.epochs_to_evaluate) :
            hist = model.fit(train_ds, epochs=1, validation_data=validation_ds, validation_freq=1)
            if hist.history['val_loss'][0] <= min_val_loss :
                min_val_loss = hist.history['val_loss'][0]
        q.put(min_val_loss)

    def evaluate_architecture(self, k, c) :
        q = Queue()
        p = Process(target=self.evaluate_architecture_process, args=(q, k, c,))
        p.start()
        p.join()
        min_val_loss = q.get()
        print(f"\n\n\nk: {k}, c: {c}, min val loss: {min_val_loss}\n\n\n")
        return min_val_loss

    def search_process(self, ) :
        k = 1

        beta = 0
        gamma = 0

        _k = 1
        min_val_loss_global = np.finfo(np.float64).max 

        while 0 < int(np.floor(2**-beta*k)) :        
            _c = 0
            min_val_loss_local = np.finfo(np.float64).max
            while self.evaluate_params_of_architecture(_k, _c) > 0 :
                _min_val_loss = self.evaluate_architecture(_k, _c)
                if _min_val_loss <= min_val_loss_local :
                    min_val_loss_local = _min_val_loss
                    c_local = _c
                    if not os.path.isfile('resulting_architecture.json') :
                        with open('resulting_architecture.json', 'w') as fp :
                            json.dump({'k': _k, 'c': _c}, fp)
                _c = _c + 1  
            
            #check if new architecture is better than old one
            if min_val_loss_local < min_val_loss_global :
                min_val_loss_global = min_val_loss_local
                k = _k
                c = c_local
                with open('new_resulting_architecture.json', 'w') as fp:
                    json.dump({'k': k, 'c': c}, fp)
                os.replace('new_resulting_architecture.json', 'resulting_architecture.json')
            else :
                gamma = 1
           
            beta = beta + gamma
            _k = k + int(np.floor(2**-beta*k))

    def search(self) :
        start_time = datetime.datetime.now()

        if os.path.isfile("resulting_architecture.json") :
            os.remove("resulting_architecture.json")
        p = Process(target=self.search_process)
        p.start()
        p.join(timeout=int(self.time_budget))

        if None == p.exitcode :
            p.terminate()
            p.join()
            print("\n\n\n\n\nProcess Terminated. Cause: resource depletion.")

        if os.path.isfile("resulting_architecture.json") :
            with open('resulting_architecture.json', 'r') as fp :
                resulting_architecture = json.load(fp)
            print("\n\n\n\n") if None != p.exitcode else None
            print(f"Resulting architecture: (k: {resulting_architecture['k']}, c: {resulting_architecture['c']}) \n\n\n\n\n")
            architecture_exists = True
        else :
            
            print("No feasible solution found.\n\n\n\n\n")
            architecture_exists = False

        if os.path.isfile('new_resulting_architecture.json') :
            os.remove('new_resulting_architecture.json')

        print(f"\nElapsed time (search): {datetime.datetime.now() - start_time}\n\n\n\n")

        return architecture_exists

    def train_resulting_architecture_process(self, batch_size, learning_rate, epochs) :
        with open('resulting_architecture.json', 'r') as fp :
            resulting_architecture = json.load(fp)
        print(f"{resulting_architecture['k']}, {resulting_architecture['c']}")
        
        train_ds, val_ds = self.load_training_and_validation_sets(batch_size)

        #quantization aware training
        model = self.model(resulting_architecture['k'], resulting_architecture['c'])[1]
        self.compile_model(model, learning_rate)

        min_val_loss = np.finfo(np.float64).max

        for i in range(epochs) :
            hist = model.fit(train_ds, epochs=1, validation_data=val_ds, validation_freq=1)
            if hist.history['val_loss'][0] <= min_val_loss :
                min_val_loss = hist.history['val_loss'][0]
                model.save("tmp.h5")
        
        model = tf.keras.models.load_model("tmp.h5")
        os.remove("tmp.h5")
        
        #full integer quantization
        #(quantization aware training returns a model 
        #using float32 for the first and the last layers
        #and int8 for the middle layers, when converted to tflite)
        self.quantize_model(model, "resulting_model.tflite", train_ds)

    def train_resulting_architecture(self, batch_size, learning_rate, epochs) : 
        with open('resulting_architecture.json', 'r') as fp :
            resulting_architecture = json.load(fp)
        self.evaluate_params_of_architecture(resulting_architecture['k'], resulting_architecture['c'])
        start_time = datetime.datetime.now()
        p = Process(target=self.train_resulting_architecture_process, args=((batch_size, learning_rate, epochs,)))
        p.start()
        p.join()
        print(f"\nElapsed time (training): {datetime.datetime.now() - start_time}\n\n\n\n")
    
    def test_tflite_model_process(self, q, path_to_test_set=None, x=None, y=None, path_to_tflite_model=None) :
        if not path_to_test_set==None :
            color_mode = self.get_color_mode()
    
            test_ds = tf.keras.utils.image_dataset_from_directory(
                directory= path_to_test_set,
                labels='inferred',
                label_mode='categorical',
                color_mode=color_mode,
                batch_size=1,
                image_size=self.input_shape[0:2],
                shuffle=True,
                seed=11,
            )
        else :
            test_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(1)
    
        interpreter = tf.lite.Interpreter("resulting_model.tflite" if None == path_to_tflite_model else path_to_tflite_model)
        interpreter.allocate_tensors()
    
        output = interpreter.get_output_details()[0]  # Model has single output.
        input = interpreter.get_input_details()[0]  # Model has single input.
    
        correct = 0
        wrong = 0
    
        for image, label in test_ds :
            # Check if the input type is quantized, then rescale input data to uint8
            if input['dtype'] == tf.uint8:
                input_scale, input_zero_point = input["quantization"]
                image = image / input_scale + input_zero_point
            input_data = tf.dtypes.cast(image, tf.uint8)
            interpreter.set_tensor(input['index'], input_data)
            interpreter.invoke()
            if label.numpy().argmax() == interpreter.get_tensor(output['index']).argmax() :
                correct = correct + 1
            else :
                wrong = wrong + 1
        print(f"\n\nTflite model test accuracy: {correct/(correct+wrong)}\n\n")
        q.put(correct/(correct+wrong))
            
    def test_tflite_model(self, path_to_test_set=None, x=None, y=None, path_to_tflite_model=None) :
        q = Queue()
        p = Process(target=self.test_tflite_model_process, args=((q, path_to_test_set, x, y, path_to_tflite_model,)))
        p.start()
        p.join()
        return q.get()
