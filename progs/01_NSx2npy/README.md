## [NPMK (Neural Processing Matlab Kit)](https://github.com/BlackrockMicrosystems/NPMK)
Raw data of our dataset are stored in .NSx format of Blackrock Microsystems (https://www.blackrockmicro.com/). In this project, NPMK is used to load raw data and save them as matlab files (*.mat).

## nsx2mat.m
ns2mat.m does the followings:
1) to load *.ns3 and *.ns4 files ("./data/orig/0?/day?/raw/0?_day?.ns?")
2) to save analog input (indicating video capturing timings) and each tetrode's voltage [uV] separately.

## nsx2mat_octave.m
The octave version of the nsx2mat.m script. The following bash one line code creates ./data/0?/day?/split_octave/2kHz_mat/*.mat files.
``` bash
octave nsx2mat_octave.m --no-gui
```

## mat2npy.py
mat2npy.py converts the format of a matrix from .mat to .npy.

## 48h_to_2days.py
48h_to_2days.py splits 48-hour recording of the mouse #05 on DAY4 into DAY4 and DAY5.

## fixme
- [ ] check if nsx2mat.m works with octave.
