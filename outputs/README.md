This folder contains the first-hand experiment results from out paper. We include the original CSV and TXT files in 
the repo. Visualization PNG files are not included due to their size. You can produce them yourself by executing the 
pipeline scripts. 

### Inside the [more_smirk_results](./more_simrk_results):

'more_smirk_evt_*_morton.csv' --> Morton code of each frame of event.

'more_smirk_evt_*_morton.png' --> The plot of the all morton codes of event. 

'more_smirk_evt_*_sfc_input.png' --> The RoI cell values of each frame 


### Inside the [pie_results](./pie_results):

'pie_003_598_690_morton.csv' 'pie_003_598_690_morton.png' 'pie_003_598_690_sfc_input.csv' 

003 --> the PIE dataset sequence ID. 

598_698 --> the start and end frame of the crossing segment. 

'pie_***_***_***_morton.csv' files are read by the [evaluation_pipeline.py](../evaluation_pipeline.py) for conclude results. 

### Inside the [pie_eval](./pie_eval):
These the results from [evaluation_pipeline.py](../evaluation_pipeline.py).
