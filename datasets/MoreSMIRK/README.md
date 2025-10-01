## MoreSmirkEvents.yml
Classifies all the crossing events in the MoreSmirk data set. The python script 'GenerateSmirkEvents.py'
reads this file to generate the time lapses of the crossing events found in the 'raw_data' folder. 

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