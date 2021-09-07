addpath('modules/buzcode/io')
addpath('modules/buzcode/externalPackages/xmltree-2.0')

basepath = 'data/th-1/data/Mouse12-120806';

% lpath_dat = 'data/th-1/data/Mouse12-120806-raw/Mouse12-120806-01.dat';
lpath_eeg = 'data/th-1/data/Mouse12-120806/Mouse12-120806.eeg';
lpath_xml = 'data/th-1/data/Mouse12-120806/Mouse12-120806.xml';
lpath_whl = 'data/th-1/data/Mouse12-120806/Mouse12-120806.whl';
% data_dat = bz_LoadBinary(lpath_dat);
data_eeg = bz_LoadBinary(lpath_eeg);
data_whl = bz_LoadBinary(lpath_whl);
data_xml = convert(xmltree(lpath_xml));

length(data_eeg) % 2.5171e+09
n_chs = 90;
samp_rate = data_xml.acquisitionSystem.samplingRate; % 20 kHz
len_per_ch = length(data_eeg) ./ n_chs
len_per_ch ./ samp_rate


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// load(lpath_dat)
// load(lpath_eeg, '-ascii')

// openNSx(lpath_eeg)
// openNSx(lpath_dat)


// addpath("modules/eeglab2021.1/")
// addpath("modules/NPMK")