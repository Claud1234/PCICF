Here we provide the two datasets, PIE and MoreSMIRk, that are used for the paper's experiments.
The raw data can be downloaded from HERE. In repository, we provide the corresponding split and detection files. 

### PIE
'pie_splits' folder contains the human annotation of pedestrians' quantity and crossing pattern in PIE dataset.  

|          Files          |                          Description                          | 
|:-----------------------:|:-------------------------------------------------------------:|
|     single_l2r.yml      |         single pedestrian crossing from left to right         | 
|     single_l2r.yml      |         single pedestrian crossing from left to right         | 
|   two_follow_l2r.yml    |     Two follow-up pedestrians crossing from left to right     | 
|   two_follow_r2l.pth    |     Two follow-up pedestrians crossing from left to right     | 
| multi_no_follow_l2r.pth | multiple no-follow-up pedestrians crossing from left to right | 
| multi_no_follow_r2l.pth | multiple no-follow-up pedestrians crossing from left to right | 
|   multi_both_dir.pth    |      multiple pedestrians crossing from both directions       |

The examples inside the yml file:
```angular2html
- id: "pie_543"
  event_window: [522, 687]
  comments: four pedestrians in two clusters with delay
```
id --> the PIE dataset sequence 

event_window --> the start and end frame of the crossing event 

comments --> human annotator's description of the crossing event. 