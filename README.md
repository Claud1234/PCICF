<div align="center">

# PCICF: A Pedestrian Crossing Identification and Classification Framework
</div>

https://github.com/user-attachments/assets/304346f5-74c2-4282-ac6a-1a9a14b2bc19

> **PCICF: A Pedestrian Crossing Identification and Classification Framework**
> - [Paper in arXiv](https://arxiv.org/abs/2509.24386)

# News
- [14-10-2025] The scripts to execute the detection and evaluation pipeline are available and verified. The PIE dataset that used to evaluate the PCICF in paper is available to download. 


## Abstract
We have recently observed the commercial roll-out of robotaxis in various countries, including the USA and Germany. They are deployed within an operational design domain (ODD) on specific routes and environmental conditions, and are subject to continuous monitoring to regain control in safety-critical traffic situations. Since ODDs typically cover urban areas, robotaxis must reliably detect and interact with vulnerable road users (VRUs) such as pedestrians, bicyclists, or e-scooter riders. To better handle such varied traffic situations, end-to-end AI, which directly compute vehicle control actions from multi-modal sensor data instead of only for perception,is on the rise. High quality data is needed for systematically training and evaluating such systems within their OOD. In this work, we propose PCICF, a framework to systematically identify and classify VRU situations to support ODD’s incident analysis. We base our work on the existing synthetic dataset SMIRK, and enhance it by extending its single-pedestrian-only design into the MoreSMIRK dataset, a structured dictionary of multi-pedestrian crossing situations constructed systematically. We then use space-filling curves (SFCs) to transform multi-dimensional features of scenarios into characteristic patterns, which we match with corresponding entries in MoreSMIRK. We evaluate PCICF with the large real-world dataset PIE, which contains more than 150 manually annotated pedestrian crossing videos. We show that PCICF can successfully identify and classify complex pedestrian crossing situations, even when groups of pedestrians merge or split during their crossing. By leveraging computationally efficient components like SFCs, PCICF has even potential to be used onboard of robotaxis for out-of-distribution (OOD) detection for example. We share an open-source replication package for PCICF containing its algorithms, the complete MoreSMIRK dataset and dictionary, as well as our experiment results.

## Method
<img width="2634" height="1373" alt="pipeline_architecture drawio" src="https://github.com/user-attachments/assets/a9b31306-f7f8-4639-ae99-ddc4e420e57b" />

## Installation
We use python 3.12 in all experiments. The pytorch version is 2.5.0, and the cuda version is 11.8.
The conda yml file of our environment is available in repo. We recommend to use this yml file when working on this repo. 

## Dataset
- [Dataset](datasets/README.md)

## Pipeline
The thorough pedestrian crossing classification was conducted by two pipeline scripts. The first is [detection_pipeline.py](detection_pipeline.py) 
that computes/plots morton codes and saves visualization results. The second is [evaluation_pipeline.py](evaluation_pipeline.py) that compare the testing 
dataset (PIE in this project) with benchmark dataset (MoreSMIRK in this project). 

The reason we separate the classification into two scripts is having more freedom on input data and parameters. 
For example, if want to extend the benchmark dataset, there is no need to run the detection on existing datasets again.   

### Detection 
The script is [detection_pipeline.py](detection_pipeline.py). There are three args for this script. \
-o --> the path to save the detection results \
-s --> save the visualization PNG results (DO NOT add this arg if don't want to save PNG result) \
-d --> dataset. Choices: pie, MoreSMIRK

In our project, for PIE dataset:

First go to [THIS LINE](https://github.com/Claud1234/MMF-TED/blob/cf247dc86886ea5dbcf88e7299551ffe0f679832/config.json#L12) in config.json to change the yaml file path for PIE input. Make sure the corresponding 
yaml file exists in the /datasets/pie_splits
```
python detection_pipeline.py -o ./outputs/pie_results/ -s -d pie
```

For MoreSMIRK dataset:
First go to [THIS LINE](https://github.com/Claud1234/MMF-TED/blob/cf247dc86886ea5dbcf88e7299551ffe0f679832/config.json#L16) in config.json to change the input path for MoreSMIRK dataset. Make sure download the raw 
data of our MoreSMIRK dataset and put them in the correct path. 
```
python detection_pipeline.py -o ./outputs/more_simrk_results/ -s -d MoreSMIRK
```

### Evaluation
First make sure running the [detection_pipeline.py](detection_pipeline.py) on both PIE and MoreSMIRK datasets, and the morton code csv files are 
saved in corresponding path in outputs folder. In our case, we have provided all csv files in the repo [outputs/pie_results](outputs/pie_results) and [outputs/more_smirk_results](outputs/more_simrk_results), so you can 
execute the commands directly. 

The script [evaluation_pipeline.py](evaluation_pipeline.py) contians two args:
-i --> the path contains all morton csv files for under-testing dataset (PIE dataset). 
-t --> the threshold of similarity between under-testing dataset and benchmark dataset (MoreSMIRK dataset).

An example for PIE dataset's single-pedestrian-crossing-from-left-to-right scenario:
```
python evaluation_pipeline.py -i ./outputs/pie_results/single_l2r -t 0.5
```

The evaluation results will be saved as txt file at [outputs/pie_eval](outputs/pie_eval)


## Output
- [Output](outputs/README.md)

## Bibtex
If anything in this repo has a use for your work, please considering to cite our work.
```
@misc{gu2025pcicfpedestriancrossingidentification,
      title={PCICF: A Pedestrian Crossing Identification and Classification Framework}, 
      author={Junyi Gu and Beatriz Cabrero-Daniel and Ali Nouri and Lydia Armini and Christian Berger},
      year={2025},
      eprint={2509.24386},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.24386}, 
}
```

