# Large Spanish EEG

EEG: silent and perceive speech on 30 spanish sentences 

Large Spanish Speech EEG [dataset](https://openneuro.org/datasets/ds004279)
Authors
<ul>
  <li>Carlos Valle</li>
  <li>Carolina Méndez-Orellana</li>
  <li>María Rodríguez-Fernández</li>
</ul>



The dataset is part of a publication by the authors: Carlos Valle, Carolina Méndez-Orellana, María Rodríguez-Fernández and Christian Herff

---
### **Identification of perceived sentences using deep neural networks in EEG**

## Abstract:
Decoding speech from brain activity can enable communication for individuals with speech disorders. Deep neural networks have shown great potential for speech decoding applications, but the large data sets required for these models are usually not available for neural recordings of speech impaired subjects. Harnessing data from other participants would thus be ideal to create speech neuroprostheses without the need of patient-specific training data.
In this study, we recorded 60 sessions from 56 healthy participants using 64 EEG channels and developed a neural network capable of subject-independent classification of perceived sentences. We found that sentence identity can be decoded from subjects without prior training achieving higher accuracy than mixed-subject models.
The development of subject-independent models eliminates the need to collect data from a target subject, reducing time and data collection costs during deployment. These results open new avenues for creating speech neuroprostheses when subjects cannot provide training data.  


## Experimental design

![](https://content.cld.iop.org/journals/1741-2552/21/5/056044/revision2/jnead88a3f1_hr.jpg)

We investigated the neural signals recorded using a 64-channel EEG system during speech perception and silent speech production tasks involving 30 daily use sentences in Spanish. The participants were instructed to listen to a spoken sentence from an audio recording and then silently repeat the sentence without any motor action.

The experimental design, a modified version of a previous study (Dash, et al), comprises four segments: rest, perception, preparation, and silent speech production. The rest segment lasted five seconds, presenting a fixation cross (+) before the stimulus onset.

During the perception section, the participants listened to the stimulus. Prior to subject S18, the perception section lasted 4 s, with each sentence being repeated 7 times. From subject S19 onward, the duration of the perception segment was increased to 5 s to match the duration of the silent speech production segment and the number of repetitions per sentence was decreased to 6 in order to maintain the overall length of the experiment. The preparation segment lasted one second and presented a blank screen, serving as a separation marker between the perception and silent speech production tasks. In the silent speech production segment lasting five seconds, subjects were instructed to silently repeat the previously heard stimulus without motor action only once. It is important to note that this study exclusively focuses on the outcomes derived from the speech perception task.



Cite:
```
@article{valle2024identification,
  title={Identification of perceived sentences using deep neural networks in EEG},
  author={Valle, Carlos and Mendez-Orellana, Carolina and Herff, Christian and Rodriguez-Fernandez, Maria},
  journal={Journal of neural engineering},
  volume={21},
  number={5},
  pages={056044},
  year={2024},
  publisher={IOP Publishing}
}
``````



# Download dataset
The dataset is hosted on [OpenNeuro](https://openneuro.org/datasets/ds004279). To download the dataset you can use the tools provided by openNeuro 

**Download from AWS S3**
```
aws s3 sync --no-sign-request s3://openneuro.org/ds004279 ds004279-download/
```

**Download using Node.js with @openneuro/cli**    <- **Recommended for reviewers**
```
openneuro download --snapshot 1.1.2 ds004279 ds004279-download/
```

**Download with DataLad**
```
datalad install https://openneuro.org/git/1/ds004279
```


## Files
To recreate the experiments, you first need to run `main_pipeline.py` inside the data folder,  followed by any of the `by_*.py`.
* `main_pipeline.py` : Contains all the details in terms of signal processing, epochs creating, artifact removal and ica decomposition. This script generates an `.npz` file that can be read in the notebooks.
* `by_*.py`: Contains the experiments for condition classification (rest, perception and production), pairs of sentences and window size effect in the decoding accuracy.

* `get_ica.py`: This script allows to re-compute the ica-weights matrix included in ica_label folder, as well as the automatic classification of the components by [ica label](https://github.com/mne-tools/mne-icalabel) mne implementation.

* `stimulus_dict.json`: Contains the transcription of the sentences used (Spanish).

## Folders
The folders included in this repository have auxiliary information that can be of interest to others.  
* `ica`: Contains pre-computed ICA weights matrix using 45 components in `.fif` format. In addition, `.json` files have the classification of the components performed by ica label mne 
* `sentences`: Original sentences presented to the volunteers in `.wav`.
* `praat_annotations`:  Contains `.TextGrid` files with word timestamps of each sentence. 

## Contact
Please contact us at this e-mail address if you have any question: cgvalle@uc.cl.


