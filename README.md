# blazeface_training
You can train blazeface on single/multi GPU machines on unlimitted samples using data generator

Dependencies: <br>
Method 1: pip install tensorflow==2.0

Method 2: 
- download this [file](https://www.dropbox.com/s/s7tyx1t4xuh8p7k/packages.zip?dl=0)
- unzip it at desired path
- change directory to the desired path
- run this command in cmd: <br>
for %x in (dir \*.*) do python -m pip install %x

<br>


**Installtion:** <br>
Install Anaconda or python from the following [link](https://www.dropbox.com/s/yurh9gu4xz3lb0x/Anaconda3-2020.02-Windows-x86_64.exe?dl=0) respectively,<br>

Training:<br>
Use single "bface_trainer_single_GPU_using_generator.py" or multi "bface_trainer_multi_GPU_using_generator.py " cpu training file to train the Blazeface based on the generators for any dataset.

