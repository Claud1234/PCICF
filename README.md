<div align="center">

# PCCF: An End-to-End Pedestrian Crossing Classification Framework to ...
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
```
 python detection_pipeline.py -o ./outputs/pie_results/ -s -d pie
```

For MoreSMIRK dataset:
```
python detection_pipeline.py -o ./outputs/more_simrk_results/ -s -d MoreSMIRK

```

## Output

## Bibtex