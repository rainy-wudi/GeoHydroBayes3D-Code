function [node, elelist, element, etype] = read_inp_mesh_fast(inp_file)
% Read an INP mesh while retaining unique element IDs.

fprintf('Reading INP file: %s\n', inp_file);

fid = fopen(inp_file, 'r');
if fid == -1
    error('Cannot open: %s', inp_file);
end

node = [];
element_map = containers.Map('KeyType', 'int32', 'ValueType', 'any');
etype = '';

in_node = false;
in_elem = false;

line_count = 0;
node_count = 0;
elem_count = 0;

while ~feof(fid)
    line = fgetl(fid);
    line_count = line_count + 1;
    
    if ~ischar(line)
        continue;
    end
    
    line = strtrim(line);
    line_upper = upper(line);
    
    % Skip blank lines and comments.
    if isempty(line) || startsWith(line, '**')
        continue;
    end
    
    % Detect the NODE keyword.
    if startsWith(line_upper, '*NODE')
        in_node = true;
        in_elem = false;
        continue;
    end
    
    % Detect the ELEMENT keyword.
    if startsWith(line_upper, '*ELEMENT')
        in_node = false;
        in_elem = true;
        
        % Extract the element type.
        if isempty(etype) && contains(line, 'type=')
            tokens = regexp(line, 'type=(\S+)', 'tokens', 'ignorecase');
            if ~isempty(tokens)
                etype = strtrim(tokens{1}{1});
                etype = regexprep(etype, ',.*', '');  % Remove content after the comma.
            end
        end
        continue;
    end
    
    % Other keywords end the current reading mode.
    if startsWith(line, '*') && ~startsWith(line, '**')
        in_node = false;
        in_elem = false;
        continue;
    end
    
    % Read node data.
    if in_node
        try
            data = str2num(line);
            if ~isempty(data) && length(data) >= 4
                node(end+1, :) = data(1:4);
                node_count = node_count + 1;
            end
        catch
            % Ignore malformed records.
        end
    end
    
    % Read element data.
    if in_elem
        try
            % Handle multiline element definitions.
            full_line = line;
            
            % Continue reading when a line ends with a comma.
            while endsWith(strtrim(full_line), ',') && ~feof(fid)
                next_line = fgetl(fid);
                line_count = line_count + 1;
                
                if ischar(next_line)
                    next_line = strtrim(next_line);
                    % Stop when the next line is a keyword.
                    if startsWith(next_line, '*')
                        fseek(fid, -length(next_line)-2, 'cof');  % Rewind.
                        break;
                    end
                    full_line = [full_line, ' ', next_line];
                else
                    break;
                end
            end
            
            % Parse element data.
            data = str2num(full_line);
            
            if ~isempty(data) && length(data) >= 2
                elem_id = data(1);
                conn = data(2:end);
                
                % Use a map to prevent duplicate element IDs.
                if ~isKey(element_map, elem_id)
                    element_map(elem_id) = conn;
                    elem_count = elem_count + 1;
                end
            end
        catch
            % Ignore malformed records.
        end
    end
end

fclose(fid);

fprintf('  Reading complete: %d lines\n', line_count);
fprintf('  Nodes: %d\n', node_count);
fprintf('  Elements: %d\n', elem_count);
fprintf('  Element type: %s\n', etype);

% Convert map entries to arrays, preserving uniqueness.
elem_ids = cell2mat(keys(element_map));
elem_ids = sort(elem_ids);

elelist = elem_ids(:);
element = cell(length(elelist), 1);

for i = 1:length(elelist)
    element{i} = element_map(elelist(i));
end

fprintf('  Final element count: %d (unique)\n', length(elelist));
fprintf('  Element ID range: [%d, %d]\n', min(elelist), max(elelist));

% Final validation.
if length(elelist) ~= length(unique(elelist))
    error('Internal error: element IDs are not unique');
end

end
