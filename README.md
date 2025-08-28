<div align="center">

# PCICF: A Pedestrian Crossing Identification and Classification Framework
</div>


## Abstract

## Method

## Installation
We use python 3.12 in all experiments. The conda yml file of our environment is available in repo. 

## Dataset
- [Dataset](datasets/README.md)

## Pipeline
The thorough pedestrian crossing classification was conducted by two pipeline scripts. The first is [detection_pipeline.py] 
that computes/plots morton codes and saves visualization results. The second is [evaluation.py] that compare the testing 
dataset (PIE in this project) with benchmark dataset (MoreSMIRK in this project). 

The reason we separate the classification into two scripts is having more freedom on input data and parameters. 
For example, if want to extend the benchmark dataset, there is no need to run the detection on existing datasets again.   

### Detection 
The script is [detection_pipeline.py]. There are three args for this script. \
-o --> the path to save the detection results \
-s --> save the visualization PNG results (DO NOT add this arg if don't want to save PNG result) \
-d --> dataset. Choices: pie, MoreSMIRK

In our project, for PIE dataset:

First go to [THIS LINE] in config.json to change the yaml file path for PIE input. Make sure the corresponding 
yaml file exists in the /datasets/pie_splits
```
python detection_pipeline.py -o ./outputs/pie_results/ -s -d pie
```

For MoreSMIRK dataset:
First go to [THIS LINE] in config.json to change the input path for MoreSMIRK dataset. Make sure download the raw 
data of our MoreSMIRK dataset and put them in the correct path. 
```
python detection_pipeline.py -o ./outputs/more_simrk_results/ -s -d MoreSMIRK
```

### Evaluation
First make sure running the detection_pipeline.py on both PIE and MoreSMIRK datasets, and the morton code csv files are 
saved in corresponding path in outputs folder. In our case, we have provided all csv files in the repo, so you can 
execute the commands directly. 

The script [evaluation_pipeline.py] contians two args:
-i --> the path contains all morton csv files for under-testing dataset (PIE dataset). 
-t --> the threshold of similarity between under-testing dataset and benchmark dataset (MoreSMIRK dataset).

An example for PIE dataset's single-pedestrian-crossing-from-left-to-right scenario:
```
python evaluation_pipeline.py -i ./outputs/pie_results/single_l2r -t 0.5
```

The evaluation results will be saved as txt file at [outputs/pie_eval]


## Output
- [Output](outputss/README.md)

## Bibtex