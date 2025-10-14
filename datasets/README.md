This is the folder contains the datasets that used in PCICF work. 

There are three datasets: MoreSMIRK, PIE, and SMIRK. The MoreSMIRK dataset is orignally created by us and serverd as a benchmarch dictionary for pedestrian crossing event classification. The PIE dataset is an open-source dataset we used to evaluate our frmaework. The SMIRK dataset is also an open-source dataset that we based to create the MoreSMIRK. 

All the raw data (PNG images) must be put in this folder in required structure if you want to execute the pipeline. Scripts read the files from this folder directly. For small files like yml, csv, etc., we include them in repo. The raw PNGs has to be downloaded from the given link and decompress here. 


### PIE
Please note in our experiments we didn't use all PIE dataset, we picked the sequences contain the pedestrian crossing and export the PNGs from the videos. 

Download the raw PNGs archive of PIE dataset from [HERE](https://www.roboticlab.eu/claude/pie/) first, then decomporess it here. You should have a folder called 'pie' here. 

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
In 'smirk' folder, it contains two baisc SMIRK sequences that we used to create the MoreSMIRK dataset. One sequence is a single pedestrain crosses from left to right, another one is from right to left. Full SMIRK datasetcan be found [HERE](https://www.ai.se/en/ai-labs/technology-infrastructure/datasets/smirk-dataset)

The SMIRK dataset is not a part of the framework pipeline, but only used to produce the MoreSMIRK dataset. We designed the MoreSMIRK to make it adapted to other applications (for instance, you need more pedestrians and change their crossing direction, speed, etc.), so we also open-source the script [GenerateEventSeq.py](../scripts/GenerateEventSeq.py) to produce the MoreSMIRK dataset. Check [README.md](../scripts/README.md) for more details. 

### MoreSMIRK 
