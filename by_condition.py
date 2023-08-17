from src.models import EEGNet
from src.dataloader import SpeechDatasetCondition
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras import utils as np_utils
import numpy as np
import parameters as p
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Experiments by condition (rest, perception and production)')
parser.add_argument('--num_runs', type=int, default=30, help="Number of repetitions")
parser.add_argument('--window', type=int, default=750, help="Window length in samples")
args = parser.parse_args()



P = defaultdict(list)

for k in tqdm(range(args.num_runs)):
     # LOAD SUBJECT DATA
     speech_dataset = SpeechDatasetCondition(p.DATASET,
                         window=args.window,
                         stims = [1, 2, 3, 4, 5], 
                         tasks= ['rest', 'perception', 'production'], 
                         subjects=p.SUBJECTS)
     
     X_train, Y_train, X_test, Y_test = speech_dataset.balanced_split(takes=1)


     # One-hot encoding of labels
     Y_train = np_utils.to_categorical(Y_train-1)
     Y_test = np_utils.to_categorical(Y_test-1)


     model = EEGNet(3, Chans = 14, Samples = args.window, 
          dropoutRate = 0.25, kernLength = 125, F1 = 8, 
          D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout')
     #model.summary()
     model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
               metrics = ['accuracy'])

     early_stopping_callbcak = tf.keras.callbacks.EarlyStopping(
     monitor='val_loss',
     min_delta=0.01,
     patience=3)


     # Train
     history = model.fit(X_train, Y_train, 
                    batch_size = 64, 
                    epochs = 200, 
                    verbose = 1, 
                    validation_data=(X_test, Y_test), 
                    callbacks=[early_stopping_callbcak])            

     y_pred = model.predict(X_test)
     y_pred = np.argmax(y_pred, axis=1) 
     y_true = np.argmax(Y_test, axis=1) 

     # Convert to python int
     y_pred = [int(i) for i in y_pred]
     y_true = [int(i) for i in y_true]

     P[f'run_{k}/y_pred'].extend(y_pred)
     P[f'run_{k}/y_true'].extend(y_true)


     np.save('results/condition/by_condition', P) 

