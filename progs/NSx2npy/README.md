## [NPMK (Neural Processing Matlab Kit)](https://github.com/BlackrockMicrosystems/NPMK)
Our raw dataset are stored in .NSx format of [Blackrock Microsystems](https://www.blackrockmicro.com/). In this project, NPMK is used to load the raw data.

## nsx2mat_matlab.m
ns2mat_matlab.m does the followings:
1) to load *.ns3 and *.ns4 files ("${SEMI_RIPPLE_HOME}/data/orig/0?/day?/raw/0?_day?.ns?")
2) to separately save analog input and each tetrode's voltage [uV]. Note that analog input indicates video capturing timings.

## nsx2mat_octave.m
The octave version of the nsx2mat_matlab.m script. The following bash one-liner creates files named ${SEMI_RIPPLE_HOME}/data/0?/day?/split_octave/2kHz_mat/*.mat.
``` bash
octave nsx2mat_octave.m --no-gui
```

## mat2npy.py
mat2npy.py converts the format of a matrix from .mat to .npy.

## 48h_to_2days.py
48h_to_2days.py splits 48-hour recording of the mouse #05 on DAY4 into DAY4 and DAY5.

## fixme
- [x] To check if nsx2mat.m works with octave.  
  - File sizes are not the same.
  - [ ] Are contents ...?

