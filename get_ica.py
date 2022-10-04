import mne
import os; os.system('clear')
from matplotlib import pyplot as plt
import json
import numpy as np
from collections import namedtuple
from mne.preprocessing import ICA
from tqdm import tqdm
import argparse
from mne_icalabel import label_components

parser = argparse.ArgumentParser(description='subjects')

parser.add_argument('-subject',type=str, default='sub-38', help="Name of the job to save it with tensorboard")

args = parser.parse_args()


## Load stimulus dict
def load_stimulus():
    # Load stimulus from stimulus_dict.json
    # The key is the index of the stimulus and the value is the text
    with open('stimulus_dict.json', 'r') as stimulus_file:
        index2stim = json.load(stimulus_file)
    return index2stim


## Step 1. Data Loading
def load_subject(subject_name, preload=False):
    # Paths for the subject
    edf_file_path = f"{subject_name}/curry/edf/{subject_name}.cdt.edf"
    edf_full_path = os.path.join(DATA_FOLDER_PATH, edf_file_path)
    
    #################################
    # Load raw data into mne format #
    #################################
    raw = mne.io.read_raw_edf(edf_full_path, stim_channel="Trigger", preload=preload, verbose=False)
    
    ################################################
    # Load segment times according to the protocol #
    ################################################
    with open('durations.json', 'r') as file:
        segment_duration_info = json.load(file)
        
    is_in_V1= subject_name in segment_duration_info['subjects']['V1']
    is_in_V2= subject_name in segment_duration_info['subjects']['V2']
    
    # Check that the subject as only one protocol associated
    assert sum([is_in_V1, is_in_V2]) == 1, 'Subjects must only have one protocol associated'
    assert not (is_in_V1 and is_in_V2), "This values must not be the same"
    
    if is_in_V1:
        segment_duration = segment_duration_info['protocol_version']['V1']
    
    if is_in_V2:
        segment_duration = segment_duration_info['protocol_version']['V2']

    #####################################
    ## load artifacts and bad channels ##
    #####################################
    time_artifacts_file = os.path.join(ARTIFACTS_FOLDER_PATH, f"{subject_name}_artifacts.txt")
    with open(time_artifacts_file, 'r') as file:
        time_artifacts = file.readlines()
        time_artifacts = time_artifacts[1:]  # Remove Header
        time_artifacts = [element.replace('\n','') for element in time_artifacts]  # Remove \n at the end
        time_artifacts = [element for element in time_artifacts if len(element)>0 ]
        time_artifacts = [element.split(',') for element in time_artifacts]  # split by ','
        time_artifacts = [(float(element[0]), float(element[1])) for element in time_artifacts]  # string to float

    # Check that the time of the artifact is correct
    # 1. The start time of an antifact must be before the end time
    # 2. the present time-stamp must be before of any of the following time stamp    
    for index, (start_arti, end_arti) in enumerate(time_artifacts):
        next_artifacts = time_artifacts[index + 1: ]
        
        # Check 1
        assert end_arti > start_arti, "The end of the artifact is before the start of it"
        
        # Check 2
        if len(next_artifacts) > 0:
            flatten_artifacts = np.array(next_artifacts).flatten()
            assert any(start_arti < flatten_artifacts), "The start artifacts is ahead of the next artifacts"
            assert any(end_arti < flatten_artifacts), "The end artifact is ahead of the next artifacts"
    
    # Bad channels
    bad_channels_file = os.path.join(ARTIFACTS_FOLDER_PATH, f"{subject_name}_badChannels.txt")
    with open(bad_channels_file, 'r') as file:
        bad_channels = file.readlines()
        bad_channels = bad_channels[1:]  # Remove Header  

    # Create artifacts data structure
    artifact_container = namedtuple('artifacts', ['time', 'channels'])
    artifacts = artifact_container(time=time_artifacts, channels=bad_channels)
    
    return raw, segment_duration, artifacts        
    
    
# Step 2. Add Montage to data
def add_montage(raw):
    # A montage is needed to perform interpolations between channels and ica

    # Set channels type
    raw.set_channel_types({'HEO':'eog', 'VEO':'eog', 'EKG':'ecg', 'EMG':'emg', 'Trigger':'stim'})

    # Load standar montage 10-05
    montage_1005 = mne.channels.make_standard_montage('standard_1005')
    
    # Change channels name to match with the ones in the montage
    old2new = {'FP1':'Fp1', 'FPZ':'Fpz','FP2': 'Fp2' , 
                'FZ': 'Fz', 'FCZ':'FCz', 'CZ':'Cz', 'CPZ':'CPz',
                'PZ':'Pz', 'POZ':'POz', 'OZ':'Oz', 'CB1':'I1', 'CB2':'I2'}
    mne.rename_channels(raw.info, old2new)
    
    # Set montage
    raw.set_montage(montage_1005, on_missing='raise')
    
    # Optional plot for channel location visualization
    if False:
        raw.plot_sensors(show_names=True, kind='3d')
        plt.show()
    
    return raw
    

# Step 3. Adding annotations for trigger events
def create_annotations(raw, segment_duration):
    # Creates annotations for each task condition
    # Each condition is marked with an id
    #  * 54: rest
    #  * 60: perception
    #  * 58: preparation
    #  * 56: production
    # The ids for each trial is marked with a number between 1 and 30
    # Others ids:
    #  * 51: Experiment start
    #  * 52: Experiment end
    #  * 53: instructions
    #  * 64: quick impedance test (not the same as the description 'Impedance Check')
    
    annot_structure = namedtuple('annotation_structure', ['orig_time', 'onset', 'description', 'duration'])
    # Filter annotations to only store values between 54, 60, 58 and 56, and the ids for the stimulus (1 - 30)
    # Also store ids values for the start and end of the recording
    annotation_list = []
    for annot in raw.annotations:
        orig_time = annot['orig_time']
        onset = annot['onset']
        description = annot['description']
        
        # Ignore impedance check trigger
        if description == 'Impedance Check':
            continue
        
        # Ignore ids for instruction
        if description == '53':
            continue
        
        annotation_list.append(
            annot_structure(
                orig_time=orig_time,
                onset=onset,
                description=description,
                duration=1
            )
        )
    
        
    # Annotations are added for each condition (rest, perception, preparation, production) with the corresponding
    # stimulus id. e.g:
    #  * rest_1 : rest condition for stimulus 1
    #  * perception_24: perception condition for stimulus 24
    orig_time_list = []
    onset_list = []
    description_list = []
    duration_list = []
    
    for index, annot in enumerate(annotation_list):
        
        if annot.description == '54':  # Rest
            stim_value = annotation_list[index + 1].description

            description_list.append(f"rest_{stim_value}")
            onset_list.append(annot.onset)
            duration_list.append(segment_duration[annot.description])
            orig_time_list.append(annot.orig_time)
            
            
        if annot.description == '60':  # Perception
            description_list.append(f"perception_{stim_value}")
            onset_list.append(annot.onset)
            duration_list.append(segment_duration[annot.description])
            orig_time_list.append(annot.orig_time)    
            
            
        if annot.description == '58':  # Preparation
            description_list.append(f"preparation_{stim_value}")
            onset_list.append(annot.onset)
            duration_list.append(segment_duration[annot.description])
            orig_time_list.append(annot.orig_time)
            
            
        if annot.description == '56':  # Production
            description_list.append(f"production_{stim_value}")
            onset_list.append(annot.onset)
            duration_list.append(segment_duration[annot.description])
            orig_time_list.append(annot.orig_time)
    
    
    
    new_annotations = mne.Annotations(onset=onset_list, duration=duration_list, description=description_list, orig_time=orig_time_list[0])

    raw.set_annotations(new_annotations)

    return raw

# Step 4. Add artifacts into the annotation variable
def add_artifacts_annotations(raw, artifacts):
    # It create BAD annotations to reject epochs
    # The artifacts are annotated by visual inspection
    
    if  len(artifacts.time) == 0:
        print("No artifacts in file")
        return raw
    
    # Save trigger annotations
    condition_annotations = raw.annotations
    
    # Using the same orig_time
    orig_time = condition_annotations[0]['orig_time']
    
    # Iterate over time artifacts and add them as bad to not use them in the analysis
    onset_list = []
    duration_list = []
    description_list = []
    
    for time_artifact in artifacts.time:
        start_time, end_time = time_artifact
        
        # Add the artifact to the annotions structure
        onset_list.append(start_time)
        duration_list.append(end_time - start_time)
        description_list.append('bad_artifact')
    
    bad_artifacts_annotations = mne.Annotations(onset=onset_list, 
                                                duration=duration_list,
                                                description=description_list,
                                                orig_time=orig_time
                                                )        

    
    raw.set_annotations(condition_annotations + bad_artifacts_annotations)
    
    
    ignore_segments = []
    _, experiment_duration = raw.get_data().shape
    experiment_duration = experiment_duration / 1000
    
    # Before experiment
    first_rest = 10**10
    for annot in raw.annotations:
        if 'rest' in annot['description']:
            if annot['onset'] < first_rest:
                first_rest = annot['onset']
    raw.set_annotations(raw.annotations + mne.Annotations(onset = 0,
                                                          duration= first_rest - 0.1,
                                                          description='bad_out',
                                                          orig_time=orig_time))
    # After experiment
    last_production = 0
    for annot in raw.annotations:
        if 'production' in annot['description']:
            if annot['onset'] > last_production:
                last_production = annot['onset'] + annot['duration']
                
    raw.set_annotations(raw.annotations + mne.Annotations(onset = last_production + 0.1,
                                                        duration= experiment_duration - last_production - 0.1,
                                                        description='bad_out',
                                                        orig_time=orig_time))

    # Breaks
    breaks = []
    for annot in raw.annotations:
        if 'production' in annot['description']:
            breaks.append((annot['description'], annot['onset']))          

    break_times = np.array([t for _, t in breaks])
    break_times_bool = np.where(np.diff(break_times) > 20)
    for t in break_times_bool[0]:
        raw.set_annotations(raw.annotations + mne.Annotations(onset = break_times[t] + 0.1 + 5,
                                                    duration= break_times[t + 1] - break_times[t] - 0.5 - 16,
                                                    description='bad_out',
                                                    orig_time=orig_time))

 
    return raw 

# Step 5. pre-processing
def pre_processing(raw, l_freq, h_freq, reference='average', only_language=True):
    # Load data
    raw.load_data()

    # Drop bad Channels
    raw.drop_channels(['EMG', 'EKG'])
    
    # Filte data
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    
    
    # Re-reference data
    if reference is not None:
        if reference == 'average':
            raw.set_eeg_reference('average', ch_type='eeg')
        else:
            raw.set_eeg_reference(reference, ch_type='eeg')
            
    # Interpolate bad channels
    #raw.interpolate_bads()

    # Pick language channels
    if only_language:
        raw.pick(['F7', 'F5', 'F3', 'FT7', 'FC5', 'T7', 'C5', 'C3', 'TP7', 'CP5', 'CP3', 'P7', 'P5', 'P3'])
    
    return raw



if __name__ == '__main__':
    # General Variables
    DATA_FOLDER_PATH = 'data'
    ARTIFACTS_FOLDER_PATH = 'annotations'  # Annotation folder
    
    # Step 0. Load aux info
    index2stim = load_stimulus()

    l_freq = 1
    h_freq = 100

    subject = args.subject
    print("subject", subject)

    # Step 1. Load data
    raw, segment_duration, artifacts = load_subject(subject, preload=False)  # triggers_sequencial

    # Step 2. Add montage to data
    raw = add_montage(raw)

    # Step 3. Create annotations for each condition
    raw = create_annotations(raw, segment_duration)
    
    # Step 4. Add artifacts to the annotations
    raw = add_artifacts_annotations(raw, artifacts)
    
    #raw.plot(block=True)    
    print(raw.annotations)
    
    ica_raw = raw.copy()
    ica_raw.load_data()
    ica_raw.filter(l_freq=1, h_freq=None)

    # Step 5. Pre-process data
    raw = pre_processing(raw, l_freq=l_freq, h_freq=h_freq, reference='average', only_language=False)

    # Step 6. Run ICA
    n_components = 45 
    ica = ICA(n_components=n_components, max_iter='auto', random_state=42, method='infomax', fit_params=dict(extended=True))
    ica.fit(ica_raw)

    ica.save(f'ica_label/{subject}-ica.fif', overwrite=True)

        
    ic_labels = label_components(raw, ica, method='iclabel')
    print(type(ic_labels))
    print(ic_labels)

    tolist_ic_labels = {
        'y_pred_proba': ic_labels['y_pred_proba'].tolist(), 
        'labels': ic_labels['labels'],
    }
    with open(os.path.join('ica_label', 'ic_labels_' + subject + '.json'), 'w') as f:
        json.dump(tolist_ic_labels, f)
