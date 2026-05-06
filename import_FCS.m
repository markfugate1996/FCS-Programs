function [T, metadata] = import_FCS()
% IMPORT_FCS
% -------------------------------------------------------------------------
% Importer for FCS text files exported from VistaVision ("MathWorks Format")
%
% File layout:
%   Lines 1–3   : Metadata in "Var = value" format
%   Line 7–end  : Numeric data
%
% First three columns are superfluous:
%   1) Record# (copy of row number)
%   2) T2record (unused)
%   3) CHN (channel artifact)
%
% Remaining four columns:
%   Ch, TrueTime, MacroTime, MicroTime
%
% Outputs:
%   T        - cleaned data table
%   metadata - struct containing metadata
% -------------------------------------------------------------------------

    %% 1) Prompt user to select file
    [fileName, filePath] = uigetfile('*.*', 'Select a file');
    if isequal(fileName,0)
        disp('User canceled file selection.');
        T = [];
        metadata = struct();
        return;
    end
    fullFileName = fullfile(filePath, fileName);

    %% 2) Determine file size
    fileInfo = dir(fullFileName);
    fileSizeMB = fileInfo.bytes / 1e6;
    largeFileThresholdMB = 500;

    %% 3) Read metadata lines ("Var = value")
    fid = fopen(fullFileName, 'r');
    if fid == -1, error('Could not open file.'); end

    metadata = struct();
    metadataLineCount = 0;

    while true
        pos = ftell(fid);
        line = fgetl(fid);
        if ~ischar(line), break; end

        parts = split(line,'=');
        if numel(parts) ~= 2
            fseek(fid, pos, 'bof'); 
            break;
        end

        varName = matlab.lang.makeValidName(strtrim(parts{1}));
        value = str2double(strtrim(parts{2}));

        if isnan(value)
            fseek(fid, pos, 'bof');
            break;
        end

        metadata.(varName) = value;
        metadataLineCount = metadataLineCount + 1;
    end
    fclose(fid);

    %% 4) Read numeric data (skip first 6 lines)
    dataStartLine = 7;

    if fileSizeMB < largeFileThresholdMB
        % Small file: read entire numeric matrix
        rawData = readmatrix(fullFileName, 'NumHeaderLines', dataStartLine-1);
        % Remove first three superfluous columns
        rawData(:,1:3) = [];
        T = array2table(rawData);
    else
        % Large file: use datastore
        fprintf('Large file detected (%.1f MB). Using datastore...\n', fileSizeMB);
        ds = tabularTextDatastore(fullFileName, 'NumHeaderLines', dataStartLine-1);
        rawTable = readall(ds);
        % Remove first three superfluous columns
        rawTable(:,1:3) = [];
        % Convert any cell columns to numeric if needed
        rawData = zeros(height(rawTable), width(rawTable));
        for k = 1:width(rawTable)
            col = rawTable{:,k};
            if iscell(col)
                rawData(:,k) = str2double(col);
            else
                rawData(:,k) = col;
            end
        end
        T = array2table(rawData);
    end

    %% 5) Assign correct column names
    T.Properties.VariableNames = {'Ch','TrueTime','MacroTime','MicroTime'};

    %% 6) Display summary
    fprintf('File successfully loaded.\n');
    fprintf('Metadata fields detected: %d\n', metadataLineCount);
    fprintf('File size: %.1f MB\n', fileSizeMB);
    fprintf('Rows loaded: %d\n', height(T));
end
