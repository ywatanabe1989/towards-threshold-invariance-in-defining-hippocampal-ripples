%#!/usr/bin/env/ octave -qf
% 5.2.0

%% Change to the Repository Home
PROJ_HOME = getenv('PROJ_HOME'); % .towards-threshold-invariance-in-defining-hippocampal-ripples
chdir(PROJ_HOME);

run("./modules/buzcode/io/bz_GetLFP")
% run('./modules/buzcode/detectors/detectEvents/bz_FindRipples.m ') 