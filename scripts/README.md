### This folder includes the python scripts to do specific tasks 

#### smirk_create_multi_pedestrian.py 
This script is making 'synthetic' multi-pedestrian crossing for smirk dataset. The smirk datasets only contains
the single-pedestrian crossing. The script reads the frame and mirror the pedestrian to another side
(horizontal flip), then save the result rgb file and corresponding annotation. 

Please note the script contains hard-embedded code that only work inside this repo. 

Make sure all the smirk frames rgb path are listed out in [all.txt](../datasets/all.txt) and the 'smirk' folder that contains all data is in [datasets](../datasets). 
Then execute:
```
cd MMM-TED/scripts
cat ../datasets/all.txt | while read line; do python smirk_create_multi_pedestrian.py $line; done;
```
It will read each line in [all.txt](../datasets/all.txt) and give it as script's argument, the script will produce the object-flipped rgb image and annotation in [datasets](../datasets) with folder name 'smirk_multi_ped'. The path hierarchy inside the 'smirk_multi_ped' will be same as 'smirk'.
