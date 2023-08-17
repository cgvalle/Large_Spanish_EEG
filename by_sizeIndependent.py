# Effects of amount of data
from src.models import EEGNet
from src.dataloader import SpeechDataset
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras import utils as np_utils
import numpy as np
import parameters as p
import argparse
from tqdm import tqdm
import random

parser = argparse.ArgumentParser(description='Experiments by condition (rest, perception and production)')
parser.add_argument('--num_runs', type=int, default=30, help="Number of repetitions")
parser.add_argument('--window', type=int, default=750, help="Window length in samples")
parser.add_argument('--task', type=str, help="Task of dataset (perception or production)")
parser.add_argument('--stims', type=int, default=5, help="Num of stims")
parser.add_argument('--shuffle', type=bool, default=False)
args = parser.parse_args()


STIMS = list(range(1,31)[:args.stims])



for num_subjects in range(2, 60):
    P = defaultdict(list)
    for iteration in tqdm(range(args.num_runs)):
        all_subjects = random.choices(p.SUBJECTS, k=num_subjects)
        test_subject = random.choices(all_subjects, k=1)
        train_subjects = [sub for sub in all_subjects if sub != test_subject[0]]


        # TRAIN SET
        speech_dataset = SpeechDataset(p.DATASET, 
                            window=args.window, 
                            shuffle_labels = args.shuffle, 
                            stims=STIMS, 
                            tasks=[args.task], 
                            subjects=train_subjects)
        X_train, Y_train, _, _ = speech_dataset.balanced_split(takes=0, shuffle_labels=args.shuffle)

        # TEST SET
        speech_dataset = SpeechDataset(p.DATASET, 
                            window=args.window, 
                            shuffle_labels = args.shuffle, 
                            stims=STIMS, 
                            tasks=[args.task], 
                            subjects=test_subject)
        X_test, Y_test, _, _= speech_dataset.balanced_split(takes=0, shuffle_labels=args.shuffle)


        # One-hot encoding of labels
        Y_train = np_utils.to_categorical(Y_train-1)
        Y_test = np_utils.to_categorical(Y_test-1)


        ### model ###

        model = EEGNet(len(STIMS), Chans = 14, Samples = args.window, 
                        dropoutRate = 0.25, kernLength = 125, F1 = 8, 
                        D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout')
        #model.summary()
        loss = 'categorical_crossentropy'
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                        metrics = ['accuracy'])


        ### Training ###
        # Early stopping
        early_stopping_callbcak = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=3
        )


        history = model.fit(X_train, Y_train, 
                            batch_size = 32, 
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

        P[f'run_{iteration}/y_pred'].extend(y_pred)
        P[f'run_{iteration}/y_true'].extend(y_true)


        #results[f"{i}_{j}"].append(history.history['val_accuracy'][-5]) 

        
        np.save(f'results/sizeIndependent/{num_subjects}_task_{args.task}_shuffle_{args.shuffle}', P) 