function [dataTables, fileNames] = import_Vista_ACFS(folderPath)
%IMPORTCSVFOLDER Import all CSV files in a folder into MATLAB tables
%
%   [dataTables, fileNames] = import_Vista_ACFS(folderPath)
%
%   - folderPath: path to the folder containing CSV files. If empty or
%                 not provided, a folder selection dialog will appear.
%   - dataTables: cell array of tables, one per CSV file.
%   - fileNames:  cell array of corresponding CSV file names.
%
%   The CSV files are expected to have a "[Data]" section. Column names
%   are hard-coded as: tau, Ch1, Ch1Err, Ch2, Ch2Err, Cross, CrossErr.

    % If folderPath not provided, prompt user
    if nargin < 1 || isempty(folderPath)
        folderPath = uigetdir(pwd, 'Select folder containing CSV files');
        if folderPath == 0
            error('No folder selected.');
        end
    end

    % List all CSV files in the folder
    csvFiles = dir(fullfile(folderPath, '*.csv'));

    % Preallocate cell arrays
    dataTables = cell(length(csvFiles), 1);
    fileNames = cell(length(csvFiles), 1);

    % Fixed column names
    colNames = {'tau','Ch1','Ch1Err','Ch2','Ch2Err','Cross','CrossErr'};

    % Loop over each CSV file
    for k = 1:length(csvFiles)
        fileName = csvFiles(k).name;
        filePath = fullfile(folderPath, fileName);
        fileNames{k} = fileName;

        % Open file
        fid = fopen(filePath, 'r');
        if fid == -1
            warning('Cannot open file: %s', fileName);
            continue;
        end

        % Read until [Data]
        line = fgetl(fid);
        while ischar(line)
            if strcmp(line, '[Data]')
                break;
            end
            line = fgetl(fid);
        end

        % Read numeric data after [Data]
        numericData = [];
        line = fgetl(fid);
        while ischar(line)
            if ~isempty(line)
                numericData = [numericData; str2double(strsplit(line, ','))]; %#ok<AGROW>
            end
            line = fgetl(fid);
        end

        fclose(fid);

        % Convert to table with column names
        tbl = array2table(numericData, 'VariableNames', colNames);
        dataTables{k} = tbl;
    end
end
