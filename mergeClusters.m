function newMatrix = mergeClusters(textFile, inputMatrix)
    % Initialize the newMatrix as a copy of the inputMatrix
    newMatrix = inputMatrix;

    % Open the text file
    fileID = fopen(textFile, 'r');
    
    % Initialize a stack of actions
    actionStack = {};
    lastTline = '';

    % Read the text file line by line
    tline = fgetl(fileID);
    while ischar(tline)
        % Skip the iteration if the current tline is identical to the last one
        if strcmp(tline, lastTline)
            tline = fgetl(fileID);
            continue;
        end
        
               % Check if the line contains the words "Merge clusters"
        if contains(tline, 'Merge clusters') 
            % Extract the original and new cluster assignments
            originalClustersStr = regexp(tline, 'Merge clusters (.*) to', 'tokens');
            newClusterStr = regexp(tline, 'to (.*?)\.', 'tokens');
            
            % Convert the comma-separated strings to arrays of numbers
            originalClusters = str2num(strrep(originalClustersStr{1}{1}, ',', ' '));
            originalClusters = originalClusters + 1;
            newCluster = str2double(newClusterStr{1}{1});
            newCluster = newCluster + 1;
            
            % Push the merge action onto the actionStack
            actionStack{end+1} = {'Merge', originalClusters, newCluster};
            lastAction = 'Merge clusters';
            
        % Check if the line contains the words "Move"
        elseif contains(tline, 'Move') 
            % Push the move action onto the actionStack
            actionStack{end+1} = {'Move'};
            lastAction = 'Move';
            
        % Check if the line contains the words "Undo cluster assign"
        elseif contains(tline, 'Undo cluster assign')
            % Pop the last action from the actionStack and reverse the action if it's a merge
            lastActionItem = actionStack{end};
            actionStack(end) = [];
            if strcmp(lastActionItem{1}, 'Merge')
                newMatrix(:, 2) = revertMerge(newMatrix(:, 2), lastActionItem{2}, lastActionItem{3});
            end
            lastAction = 'Undo cluster assign';
            
        % Check if the line contains the words "Redo"
        elseif contains(tline, 'Redo') && ~strcmp(lastAction, 'Redo')
            % Push the last popped action back onto the actionStack
            actionStack{end+1} = lastActionItem;
            lastAction = 'Redo';
        else
            lastAction = '';
        end
        
        % Read the next line
        tline = fgetl(fileID);
    end
    
    % Apply all merge actions from the actionStack to the newMatrix
    for i = 1:length(actionStack)
        action = actionStack{i};
        if strcmp(action{1}, 'Merge')
            originalClusters = action{2};
            newCluster = action{3};
            for j = 1:size(newMatrix, 1)
                if ismember(newMatrix(j, 2), originalClusters)
                    newMatrix(j, 2) = newCluster;
                end
            end
        end
    end
    
    % Close the text file
    fclose(fileID);
end

% Helper function to revert a merge action
function revertedClusters = revertMerge(clusters, originalClusters, newCluster)
    revertedClusters = clusters;
    for i = 1:length(originalClusters)
        revertedClusters(clusters == newCluster) = originalClusters(i);
    end
end
   