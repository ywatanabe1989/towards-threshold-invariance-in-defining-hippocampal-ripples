%#!/usr/bin/env/ octave -qf
% 5.2.0

%% Change to the Repository Home
PROJ_HOME = getenv('PROJ_HOME'); % .towards-threshold-invariance-in-defining-hippocampal-ripples
chdir(PROJ_HOME);


%% Install NPMK
% run("./progs/01_NSx2npy/NPMK/installNPMK")
run("./modules/NPMK/installNPMK");


%% Mouse #01
LOADPATHS_MOUSE_01 = {
"./data/okada/orig/01/day1/raw/01_day1.ns4",
"./data/okada/orig/01/day1/raw/01_day1.ns3",
};

for i_f = 1:length(LOADPATHS_MOUSE_01)
    % PATHs
    loadpath = char(LOADPATHS_MOUSE_01(i_f));
    [dirname_raw, fname, ext] = fileparts(loadpath);
    dirname_split = strsplit(dirname_raw, "orig/");
    dirname_split = strcat(dirname_split{1,1}, dirname_split{1,2});
    dirname_split = erase(dirname_split, "/raw");
    dirname_split = strcat(dirname_split, "/split_octave/LFP_MEP_2kHz_mat/");

    % analog input, or video capturing timings
    if i_f == 1        
        % Load
        NS4 = openNSx(loadpath, 'precision', 's:5'); % 'precision');
        disp(strcat('Loaded: ', loadpath))

        % Save
        save_data = NS4.Data;        
        save_fname = 'analog_input.mat';
        mkdir(dirname_split);        
        savepath = strcat(dirname_split, save_fname);
        save('-v7', savepath, 'save_data')
        disp(strcat('Saved to: ', savepath))
    end

    % LFP or MEP
    if i_f == 2        
        % Load
        NS3 = openNSx(loadpath, 'uV', 'precision'); % fixme
        disp(strcat('Loaded: ', loadpath))

        % Save                                
        for i_ns3 = 1:size(NS3.Data, 1)
            save_data = NS3.Data(i_ns3,:);            
            label = NS3.ElectrodesInfo(1, i_ns3).Label;
            save_fname = strcat(label(1:5), ".mat");
            mkdir(dirname_split);            
            savepath = strcat(dirname_split, save_fname);
            save('-v7', savepath, 'save_data')
            disp(strcat('Saved to: ', savepath))
        end
    end
end
clear



%% Mouse #02, #03, #04, and "#05 on DAY 1, 2, 3"
LOADPATHS_MOUSE_02_03_04_05 = {
"./data/okada/orig/02/day1/raw/02_day1.ns3",
"./data/okada/orig/02/day2/raw/02_day2.ns3",
"./data/okada/orig/02/day3/raw/02_day3.ns3",
"./data/okada/orig/02/day4/raw/02_day4.ns3",
"./data/okada/orig/03/day1/raw/03_day1.ns3",
"./data/okada/orig/03/day2/raw/03_day2.ns3",
"./data/okada/orig/03/day3/raw/03_day3.ns3",
"./data/okada/orig/03/day4/raw/03_day4.ns3",
"./data/okada/orig/04/day1/raw/04_day1.ns3",
"./data/okada/orig/04/day2/raw/04_day2.ns3",
"./data/okada/orig/04/day3/raw/04_day3.ns3",
"./data/okada/orig/04/day4/raw/04_day4.ns3",
"./data/okada/orig/05/day1/raw/05_day1.ns3",
"./data/okada/orig/05/day2/raw/05_day2.ns3",
"./data/okada/orig/05/day3/raw/05_day3.ns3",
"./data/okada/orig/05/day4/raw/05_day4.ns3",
};

for i_f = 1:length(LOADPATHS_MOUSE_02_03_04_05)
    % PATHs
    loadpath = char(LOADPATHS_MOUSE_02_03_04_05(i_f));
    [dirname_raw, fname, ext] = fileparts(loadpath);
    dirname_split = strsplit(dirname_raw, "orig/");
    dirname_split = strcat(dirname_split{1,1}, dirname_split{1,2});
    dirname_split = erase(dirname_split, "/raw");
    dirname_split = strcat(dirname_split, "/split_octave/LFP_MEP_2kHz_mat/");

    % Load
    NS3 = openNSx(loadpath, 'uV', 'precision');
    disp(strcat('Loaded: ', loadpath))
    data_size = size(NS3.Data, 1);

    % Save
    for i_label = 1:data_size
        
        % LFP or MEP
        if i_label < data_size 
            label = NS3.ElectrodesInfo(1, i_label).Label;
            save_fname = strcat(label(1:5), ".mat");
            save_data = NS3.Data(i_label,:);
        end

        % analog input, or video capturing timings
        if i_label == data_size
            % label = NS3.ElectrodesInfo(1, i_label).Label;
            % save_fname = strcat(label(1:5), ".mat"); % analog_input.mat
            
            save_fname = 'analog_input.mat';
            save_data = (NS3.Data(i_label,:) ./ 4) * 6.5534;
        end

        % Save
        savepath = strcat(dirname_split, save_fname);            
        mkdir(dirname_split);
        save('-v7', savepath, 'save_data')
        disp(strcat('Saved to: ', savepath))
        
    end
end
clear


%% Mouse #05 day4
% PATH
loadpath = './data/okada/orig/05/day4/raw/05_day4.ns3';
[dirname_raw, fname, ext] = fileparts(loadpath);
dirname_split = strsplit(dirname_raw, "orig/");
dirname_split = strcat(dirname_split{1,1}, dirname_split{1,2});
dirname_split = erase(dirname_split, "/raw");
dirname_split = strcat(dirname_split, "/split_octave/LFP_MEP_2kHz_mat/");
% Load
NS3 = openNSx(loadpath, 'uV', 'precision');
disp(strcat('Loaded: ', loadpath))
data1 = NS3.Data{1,1};
data2 = NS3.Data{1,2};
data_size = size(data1, 1); % data1_size and data2_size are the same.
% Save
% LFP or MEP
for i_label = 1:data_size-1 
    label = NS3.ElectrodesInfo(1, i_label).Label;
    save_fname = strcat(label(1:5), ".mat");    
    savepath1 = strcat(dirname_split, "1_", save_fname);
    savepath2 = strcat(dirname_split, "2_", save_fname);    

    mkdir(dirname_split);
    
    save_data = data1(i_label,:);
    save('-v7', savepath1, 'save_data')
    disp(strcat('Saved to: ', savepath1))
    
    save_data = data2(i_label,:);
    save('-v7', savepath2, 'save_data')
    disp(strcat('Saved to: ', savepath2))
end
% analog input, or video capturing timings
label = 'analog_input';
save_fname = strcat(label, ".mat");
savepath1 = strcat(dirname_split, "1_", save_fname);
savepath2 = strcat(dirname_split, "2_", save_fname);

save_data = (data1(data_size,:) ./ 4) * 6.5534;
save('-v7', savepath1, 'save_data')
disp(strcat('Saved to: ', savepath1))

save_data = (data2(data_size,:) ./ 4) * 6.5534;
save('-v7', savepath2, 'save_data')
disp(strcat('Saved to: ', savepath2))

%% EOF