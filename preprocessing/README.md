## [NPMK (Neural Processing Matlab Kit)](https://github.com/BlackrockMicrosystems/NPMK)
Our raw dataset are stored in .NSx format of [Blackrock Microsystems](https://www.blackrockmicro.com/). In this project, NPMK is used to load the raw data.


## nsx2mat_matlab.m
This script does the followings:
1. To load *.ns3 and *.ns4 files ("${SEMI_RIPPLE_HOME}/data/orig/0?/day?/raw/0?_day?.ns?")
2. To separately save analog input and each tetrode's voltage [uV]. Note that analog input indicates video capturing timings.


## nsx2mat_octave.m
This script is the octave version of the nsx2mat_matlab.m script. The following bash one-liner creates files named "${SEMI_RIPPLE_HOME}/data/0?/day?/split_octave/LFP_MEP_2kHz_mat/*tt?-?.mat."
``` bash
$ octave nsx2mat_octave.m --no-gui
```

## mat_check.py
This script confirms the *.mat file contents originated from **./nsx2mat_matlab.m** and **./nsx2mat_octave.m** are the same.


## mat2npy.py
This script does the followings:
1. To load a "${SEMI_RIPPLE_HOME}/data/0?/day?/split_octave/LFP_MEP_2kHz_mat/*tt?-?.mat" file.
2. To down-sample the loaded 1D LFP (local field potential) or MEP (myoelectric potential) data.
3. To save the down-sampled 1D signal as a numpy file (e.g., ./data/0?/day?/split/1kHz_npy/*tt?-?_fp16.npy).


## mat2npy.sh
This script globs ./data/0?/day?/split/LFP_MEP_2kHz_mat/*tt?-?.mat" file paths and run **./mat2npy.py** on each of them.


## mouse05_48h_to_2days.py
This script splits about 48-hour recording of the mouse #05 on "DAY4" into DAY4 and DAY5.


## mouse05_48h_to_2days.sh
This script scales **./48h_to_2days.py**.


## creates_fpaths_list_of_hippo_LFP_and_trape_MEP_tts.sh
This script creates file path list of hippocampal LFP ($SEMI_RIPPLE_HOME/data/FPATH_LISTS/HIPPO_LFP_TT_IDs.txt, the list of 184 numpy files) and trapezius MEP ($SEMI_RIPPLE_HOME/data/FPATH_LISTS/TRAPE_MEP_TT_IDs.txt, the list of 36 numpy files).
