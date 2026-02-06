This is the folder contains the datasets that used in PCICF work. 

There are three datasets: MoreSMIRK, PIE, and SMIRK. The MoreSMIRK dataset is originally created by us and serverd as a benchmark dictionary for pedestrian crossing event classification. The PIE dataset is an open-source dataset we used to evaluate our framework. The SMIRK dataset is also an open-source dataset that we based to create the MoreSMIRK. 

All the raw data (PNG images) must be put in this folder in required structure if you want to execute the pipeline. Scripts read the files from this folder directly. For small files like yml, csv, etc., we include them in repo. The raw PNGs has to be downloaded from the given link and decompress here. 


### PIE
Please note in our experiments we didn't use all PIE dataset, we picked the sequences contain the pedestrian crossing and export the PNGs from the videos. 

Unfortunately the public link to download the PIE dataset (around 15 GB) that used in this project in no longer available, please contact 
claude.gujunyi@gmail.com if you are interested in it, we can figure out a way to transfer the dataset. 

When you get the dataset archive, decompress it here. You should have a folder called 'pie'. 

'pie_splits' folder contains the human annotation of pedestrian crossing in PIE dataset. When execute the [detection_pipeline.py](../detection_pipeline.py), you have to specify one of the yml files in config.json. 

|          Files          |                          Description                          | 
|:-----------------------:|:-------------------------------------------------------------:|
|     single_l2r.yml      |         single pedestrian crossing from left to right         | 
|     single_l2r.yml      |         single pedestrian crossing from left to right         | 
|   multi_follow_l2r.yml    |     Multiple follow-up pedestrians crossing from left to right     | 
|   multi_follow_r2l.yml    |     Multiple follow-up pedestrians crossing from left to right     | 
| multi_no_follow_l2r.yml | multiple no-follow-up pedestrians crossing from left to right | 
| multi_no_follow_r2l.yml | multiple no-follow-up pedestrians crossing from left to right | 
|   multi_both_dir.yml    |      multiple pedestrians crossing from both directions       |

The examples inside the yml file:
```angular2html
- id: "pie_543"
  event_window: [522, 687]
  comments: four pedestrians in two clusters with delay
```
id --> the PIE dataset sequence 

event_window --> the start and end frame of the crossing event 

comments --> human annotator's description of the crossing event. 

### SMIRK
In 'smirk' folder, it contains two basic SMIRK sequences that we used to create the MoreSMIRK dataset. One sequence is a single pedestrain crosses from left to right, another one is from right to left. Full SMIRK datasetcan be found [HERE](https://www.ai.se/en/ai-labs/technology-infrastructure/datasets/smirk-dataset)

The SMIRK dataset is not a part of the framework pipeline, but only used to produce the MoreSMIRK dataset. We designed the MoreSMIRK to make it adapted to other applications (for instance, you need more pedestrians and change their crossing direction, speed, etc.), so we also open-source the script [GenerateEventSeq.py](../scripts/GenerateEventSeq.py) to produce the MoreSMIRK dataset. Check [README.md](../scripts/README.md) for more details. 

### MoreSMIRK 
The MoreSMIRK dataset contains a total of 104 sequences. Each sequence contains 100 frames and represents a unique 
pedestrian crossing event. Thus, it can serve as a dictionary for pedestrian crossing classification. 
Please refer to Section 3.1 in our paper for more details of the MoreSMIRK dataset. 

We provide script [GenerateEventSeq.py](../scripts/GenerateEventSeq.py) to produce the MoreSMIRK dataset yourself.
The raw data that we used in paper is available at [AI Sweden](). Download/produce the dataset and have a folder 'raw_data' in 
'MoreSMIRK'. 

In our paper, we compute the Morton codes for each MoreSMIRK sequence, and use the Morton codes as the dictionary
reference. We open-source the morton codes [CSVs](./MoreSMIRK/morton_codes) in this repo as well. These CSVs are used in the framework pipeline 
to classify the pedestrian crossing. 

For [MoreSmirkEvents.yml](./MoreSMIRK/MoreSmirkEvents.yml), it used by script [GenerateEventSeq.py](../scripts/GenerateEventSeq.py) 
to generate the time lapses of the crossing events found in the 'raw_data' folder. You can modify this yml file if 
you want to create your own sequence. 

The examples inside the MoreSmirkEvents.yml file:
```
- event: 4
  pedestrians: [0, 0, 0, 1, 0, 0]
  offset: 0
```
event :  
The id of the crossing event.

pedestrians :  
The pedestrians which are active (1) or inactive (0) in the crossing event.
The three first elements represent the cluster of left (L) to right (R) walking 
pedestrians and the three last represent the cluster of R to L walking pedestrians.

offset :  
The offset of the L to R walking pedestrians. An offset of 1 shifts the L to R 
walking pedestrians by 60 px to the left.