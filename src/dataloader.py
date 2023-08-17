from collections import namedtuple, defaultdict
import numpy as np
from tqdm import tqdm
import random

trial_info = namedtuple('trial_structure', ['subject', 'task', 'stim', 'epoch', 'label'])
eeg_load_trial = namedtuple('eeg_preload_trial', ['data', 'label'])

#######################################################
#######################################################
#######################################################
class SpeechDatasetCondition:

    def __init__(self, filename, window, subjects=None, tasks=None, stims=None, trials=None):
        self.filename = filename
        self.subjects = subjects
        self.tasks = tasks
        self.tasks_id = {task_name: 1+index for index, task_name in enumerate(tasks)}
        self.stims = stims
        self.srate = 1000
        self.window = window
        
    
        

        if trials is None:
            self.stim2id = {stim_value: index+1 for index, stim_value in enumerate(self.stims)}
            self.get_info()
        else:
            self.trials = trials



    def get_info(self):
        self.trials = []  # Trials will be stored here

        with np.load(self.filename, 'r') as data:
            print("Loading subjects info")
            # Iterate over subjects, conditions and stims
            for key in tqdm(data.keys()):
                subject, task, stim = key.split('_')
                stim = int(stim)  # convert to int

                # Filter by subject, task and stim according to the inputs 
                # * Subject
                if self.subjects is not None:
                    if subject not in self.subjects:  # If subject not in self.subject continue
                        continue
                
                # * Task
                if self.tasks is not None:
                    if task not in self.tasks:
                        continue
                
                # * stim
                if self.stims is not None:
                    if stim not in self.stims:
                        continue

                # Iterate over repetitions for a stimulus
                num_epoch, _, _ = data[key].shape
                for epoch_index in range(num_epoch):
                    self.trials.append(
                        trial_info(
                            subject=subject,
                            task=task,
                            stim=stim,
                            label=self.stim2id[stim],
                            epoch=epoch_index,
                        )
                    )

    def load_eeg(self):
        self.X = np.zeros((len(self.trials), 14, self.window, 1))
        self.Y = np.zeros((len(self.trials)))
        

        with np.load(self.filename) as data:
            print("Loading eegdata into memory")
            for idx, single_trial in tqdm(enumerate(self.trials), total=len(self.trials)):
                subject = single_trial.subject
                task = single_trial.task
                stim = single_trial.stim
                epoch = single_trial.epoch
                label = single_trial.label

                text =  f"{subject}_{task}_{stim}"
                self.X[idx, :, :, 0] = data[text][epoch][:, :self.window][:, :]  # Only 3 seconds
                self.Y[idx] = self.tasks_id[task]
             

        return self.X, self.Y        
    

    def balanced_split(self, takes=2):
        # Split sets into train and test
        
        to_split = defaultdict(list)
        
        for trial in self.trials:
            subject = trial.subject
            task = trial.subject
            stim = trial.stim
            
            to_split[f"{subject}_{task}_{stim}"].append(trial)
        
        train = []
        test = []
        
        for trial_list in to_split.values():
            indexs = list(range(len(trial_list)))
            
            test_index = random.sample(indexs, takes)
            train_index = [i for i in indexs if i not in test_index]

            
            # add for test
            for t in test_index:
                test.append(trial_list[t])
                
            # add for train
            for t in train_index:
                train.append(trial_list[t])
        
            
        train_dataset = SpeechDatasetCondition(self.filename, trials=train, tasks=self.tasks,  window=self.window)
        test_dataset = SpeechDatasetCondition(self.filename, trials=test, tasks=self.tasks, window=self.window)
                    
        X_train, Y_train = train_dataset.load_eeg()
        X_test, Y_test = test_dataset.load_eeg()
    
        
        return X_train, Y_train, X_test, Y_test


#######################################################
#######################################################
#######################################################


class SpeechDataset:

    def __init__(self, filename, window, shuffle_labels=False, subjects=None, tasks=None, stims=None, trials=None):
        self.filename = filename
        self.subjects = subjects
        self.tasks = tasks
        self.stims = stims
        self.window = window
        self.shuffle_labels = shuffle_labels

        

        if trials is None:
            self.stim2id = {stim_value: index+1 for index, stim_value in enumerate(self.stims)}
            self.get_info()

        else:
            self.trials = trials



    def get_info(self):
        self.trials = []  # Trials will be stored here

        with np.load(self.filename, 'r') as data:
            print("Loading subjects info")
            # Iterate over subjects, conditions and stims
            for key in tqdm(data.keys()):
                subject, task, stim = key.split('_')
                stim = int(stim)  # convert to int

                # Filter by subject, task and stim according to the inputs 
                # * Subject
                if self.subjects is not None:
                    if subject not in self.subjects:  # If subject not in self.subject continue
                        continue
                
                # * Task
                if self.tasks is not None:
                    if task not in self.tasks:
                        continue
                
                # * stim
                if self.stims is not None:
                    if stim not in self.stims:
                        continue

                # Iterate over repetitions for a stimulus
                num_epoch, _, _ = data[key].shape
                for epoch_index in range(num_epoch):
                    self.trials.append(
                        trial_info(
                            subject=subject,
                            task=task,
                            stim=stim,
                            label=self.stim2id[stim],
                            epoch=epoch_index,
                        )
                    )

    def load_eeg(self):
        self.X = np.zeros((len(self.trials), 14, self.window, 1))
        self.Y = np.zeros((len(self.trials)))
        

        with np.load(self.filename) as data:
            print("Loading eegdata into memory")
            for idx, single_trial in tqdm(enumerate(self.trials), total=len(self.trials)):
                subject = single_trial.subject
                task = single_trial.task
                stim = single_trial.stim
                epoch = single_trial.epoch
                label = single_trial.label

                text =  f"{subject}_{task}_{stim}"
                self.X[idx, :, :, 0] = data[text][epoch][:, :self.window][:, :]  # Only 3 seconds
                if self.shuffle_labels:
                    self.Y[idx] = random.randint(1, len(self.stims))
                else:
                    self.Y[idx] = label
             

        return self.X, self.Y        

    def balanced_split(self, takes=2, shuffle_labels=False):
        # Split sets into train and test
        
        to_split = defaultdict(list)
        
        for trial in self.trials:
            subject = trial.subject
            task = trial.subject
            stim = trial.stim
            
            to_split[f"{subject}_{task}_{stim}"].append(trial)
        
        train = []
        test = []
        
        for trial_list in to_split.values():
            indexs = list(range(len(trial_list)))
            
            test_index = random.sample(indexs, takes)
            train_index = [i for i in indexs if i not in test_index]

            
            # add for test
            for t in test_index:
                test.append(trial_list[t])
                
            # add for train
            for t in train_index:
                train.append(trial_list[t])
        
            
        train_dataset = SpeechDataset(self.filename, self.window, shuffle_labels=shuffle_labels,stims=self.stims, trials=train)
        test_dataset = SpeechDataset(self.filename, self.window, shuffle_labels=shuffle_labels, stims=self.stims, trials=test)
                    
        X_train, Y_train = train_dataset.load_eeg()
        X_test, Y_test = test_dataset.load_eeg()
    
        
        return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    speech = SpeechDataset(
        filename = 'data/npz/language_average_2-50hz_icaLabel95confidence_eyes_60sessions.npz',
        window=750,
        stims=[1,2,3,4,5],
        tasks=['perception'],
        subjects= ['S57']
    )

    X_train, Y_train, _, _ = speech.balanced_split(takes=0, shuffle_labels=False)
    print(X_train.shape)
