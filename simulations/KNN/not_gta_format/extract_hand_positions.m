% function hand_positions = extract_hand_positions(trial, idx, coord)
%     num_trials = length(idx);
%     hand_positions = zeros(num_trials, 1);

%     for i = 1:num_trials
%         trial_num = idx(i); % Get trial index
%         angle_num = mod(i-1, size(trial, 2)) + 1; % Extract correct angle index

%         if coord == 'x'
%             hand_positions(i) = trial(trial_num, angle_num).handPos(1, end); % Last X position
%         else
%             hand_positions(i) = trial(trial_num, angle_num).handPos(2, end); % Last Y position
%         end
%     end
% end

function hand_positions = extract_hand_positions(trial, num_trials, num_angles, coord)
    % Extract hand positions in the same order as pop_activity
    % Inputs:
    %   trial       - Original trial structure (100x8)
    %   num_trials  - Number of trials per angle
    %   num_angles  - Number of movement angles
    %   coord       - 'x' or 'y' for extracting respective hand positions
    % Output:
    %   hand_positions - Vector (num_trials * num_angles, 1) matching pop_activity

    total_trials = num_trials * num_angles; % Matches pop_activity shape
    hand_positions = zeros(total_trials, 1); % Initialize output

    trial_idx = 1; % Index to match pop_activity order
    for angle_num = 1:num_angles
        for trial_num = 1:num_trials
            % Extract hand position (X or Y) at the last recorded time point
            if coord == 'x'
                hand_positions(trial_idx) = trial(trial_num, angle_num).handPos(1, end);
            else
                hand_positions(trial_idx) = trial(trial_num, angle_num).handPos(2, end);
            end
            trial_idx = trial_idx + 1;
        end
    end
end
