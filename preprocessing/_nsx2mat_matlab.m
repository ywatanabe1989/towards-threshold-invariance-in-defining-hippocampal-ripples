cd /mnt/md0/okada-san/progs


%% mouse #01
loadpaths_list = [
"../data/01/day1/raw/01_day1.ns4",
"../data/01/day1/raw/01_day1.ns3",
];

for f = 1:length(loadpaths_list) % fixme
    loadpath = char(loadpaths_list(f));
    [dirname_raw, fname, ext] = fileparts(loadpath);
    dirname_split = erase(dirname_raw, "/raw") + "/split/";

    if f == 1 % mouse 01, analog_input
        % NS4 = openNSx(loadpath, 'precision'); % analog input
        NS4 = openNSx(loadpath, 'precision', 's:5'); % 'precision'); % analog input)
        disp('Loaded ' + string(loadpath))
        save_data = NS4.Data;
        label = 'analog_input'; % NS4.ElectrodesInfo(1, i).Label;
        save_fname = label + ".mat";
        savepath = dirname_split + save_fname;
        % save_data = data;
        save(savepath, 'save_data')
        disp('Saved ' + savepath)
    end

    if f == 2 % mouse 01, data
        NS3 = openNSx(loadpath, 'uV', 'precision');
        disp('Loaded ' + string(loadpath))
        for i = 1:size(NS3.Data, 1)
             label = NS3.ElectrodesInfo(1, i).Label;
             save_fname = label(1:5) + ".mat";
             savepath = dirname_split + save_fname;
             save_data = NS3.Data(i,:);
             save(savepath, 'save_data')
             disp('Saved ' + savepath)
        end
    end
end


%% mouse #02, #03, #04, (#05 day1,2,3)    
loadpaths_list = [
"../data/02/day1/raw/02_day1.ns3",
"../data/02/day2/raw/02_day2.ns3",
"../data/02/day3/raw/02_day3.ns3",
"../data/02/day4/raw/02_day4.ns3",
"../data/03/day1/raw/03_day1.ns3",
"../data/03/day2/raw/03_day2.ns3",
"../data/03/day3/raw/03_day3.ns3",
"../data/03/day4/raw/03_day4.ns3",
"../data/04/day1/raw/04_day1.ns3",
"../data/04/day2/raw/04_day2.ns3",
"../data/04/day3/raw/04_day3.ns3",
"../data/04/day4/raw/04_day4.ns3",
"../data/05/day1/raw/05_day1.ns3",
"../data/05/day2/raw/05_day2.ns3",
"../data/05/day3/raw/05_day3.ns3",
"../data/05/day4/raw/05_day4.ns3",
];

for f = 1:length(loadpaths_list)
    loadpath = char(loadpaths_list(f));
    [dirname_raw, fname, ext] = fileparts(loadpath);
    dirname_split = erase(dirname_raw, "/raw") + "/split/";
    
    NS3 = openNSx(loadpath, 'uV', 'precision'); % data
    disp('Loaded ' + string(loadpath))
    data_size = size(NS3.Data, 1);
    for i = 1:data_size
        if i < data_size % data
            label = NS3.ElectrodesInfo(1, i).Label;
            save_fname = label(1:5) + ".mat";
            savepath = dirname_split + save_fname;
            save_data = NS3.Data(i,:);
            save(savepath, 'save_data')
            disp('Saved ' + savepath)
        end

        if i == data_size % analog_input
            NS3 = openNSx(loadpath, 'uV', 'precision');
            disp('Loaded ' + string(loadpath) + ' again for analog input')
            data = NS3.Data;
            label = 'analog_input';
            save_fname = label + ".mat";
            savepath = dirname_split + save_fname;
            save_data = (NS3.Data(i,:) ./ 4) * 6.5534; % fixme: 'uV', 'precision', ./4, * 6.5534
            save(savepath, 'save_data')
            disp('Saved ' + savepath)
        end
    end
end


%% mouse #05 day4
% data
loadpath = '../data/05/day4/raw/05_day4.ns3';
[dirname_raw, fname, ext] = fileparts(loadpath);
dirname_split = erase(dirname_raw, "/raw") + "/split/";
NS3 = openNSx(loadpath, 'uV', 'precision');
disp('Loaded ' + string(loadpath))
data1 = NS3.Data{1,1};
data2 = NS3.Data{1,2};
data_size = size(data1, 1); % data1_size == data2_size
for i = 1:data_size-1 % data
    label = NS3.ElectrodesInfo(1, i).Label;
    save_fname = label(1:5) + ".mat";
    savepath1 = dirname_split + "1_" + save_fname;
    savepath2 = dirname_split + "2_" + save_fname;
    save_data = data1(i,:);
    save(savepath1, 'save_data')
    disp('Saved ' + savepath1)
    save_data = data2(i,:);
    save(savepath2, 'save_data')
    disp('Saved ' + savepath2)
end
% analog_input

NS3 = openNSx(loadpath, 'uV', 'precision');
disp('Loaded ' + string(loadpath) + ' again for analog input')
data1 = NS3.Data{1,1};
data2 = NS3.Data{1,2};

label = 'analog_input';
save_fname = label + ".mat";
savepath1 = dirname_split + "1_" + save_fname;
savepath2 = dirname_split + "2_" + save_fname;
save_data = (data1(data_size,:) ./ 4) * 6.5534;
save(savepath1, 'save_data')
disp('Saved ' + savepath1)
save_data = (data2(data_size,:) ./ 4) * 6.5534;
save(savepath2, 'save_data')
disp('Saved ' + savepath2)


%% Checking
%{
cd ../../../05/day4/split/

name = "analog_input";
orig_ai = load('./orig/' + name + '.mat');
made_ai = load('./' + name + '.mat');
size(orig_ai.analog_input)
size(made_ai.save_data)
transpose(orig_ai.analog_input(1:10)) ./ double(made_ai.save_data(1:10))

tt = "tt8";
num = "1";
orig_tt = load('./orig/' + tt + "_" + num + '.mat');
made_tt = load('./' + tt + "-" + num + '.mat');
size(orig_tt.tt8_1)
size(made_tt.save_data)
transpose(orig_tt.tt8_1(1:10)) ./ double(made_tt.save_data(1:10))
%}

%{
% 01
openNSx % NS4, analog input
disp(NS4.ElectrodeInfo.Label) % ainp1
ainp1 = NS4.Data;


%   'uV':         Will read the spike waveforms in unit of uV instead of
%                 raw values. Note that this conversion may lead to loss of
%                 information (e.g. 15/4 = 4) since the waveforms type will
%                 stay in int16. It's recommended to read raw spike
%                 waveforms and then perform the conversion at a later
%                 time.
%                 DEFAULT: will read waveform information in raw.

loadpath = "../data/01/day1/raw/01_day1.ns3";
[dirname_raw, fname, ext] = fileparts(loadpath);
dirname_split = erase(dirname_raw, "/raw") + "/split/";

data = openNSx(loadpath); % NS4, analog input
disp('Loaded ' + string(loadpath))

for i = 1:size(data.Data, 1)
    label = data.ElectrodesInfo(1, i).Label;
    % label = label(1:5);
    save_fname = label(1:5) + ".mat";
    savepath = dirname_split + save_fname;
    save_data = data.Data(i,:);
    save(savepath, 'save_data')
    disp('Saved ' + savepath)
end


    disp(NS3.ElectrodesInfo(1,i).Label
    label = NS3.ElectrodeInfo.Label(i);
    save(label, NS3.Data(i))
    disp

for i = 1:19;
    row = NS3.Data(i,:);
    save("tt" + string(i), "row")
end

cell1 = NS3.Data(1,1);
cell2 = NS3.Data(1,2);

NS3_1 = cell1{1};
NS3_2 = cell2{1};

for i = 1:23;
    row = NS3_2(i,:);
    save("2_tt" + string(i), "row")
end
%}