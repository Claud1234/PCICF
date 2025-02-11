### This folder includes the python scripts to do specific tasks 

#### smirk_create_multi_pedestrian.py 
This script is making 'synthetic' multi-pedestrian crossing for smirk dataset. The smirk datasets only contains
the single-pedestrian crossing. The script reads the frame and mirror the pedestrian to another side
(horizontal flip), then save the result rgb file and corresponding annotation. 

Please note the script contains hard-embedded code that only work inside this repo. 

Make sure all the smirk frames rgb path are listed out in 