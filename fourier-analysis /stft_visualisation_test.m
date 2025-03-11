% Load the training data (ensure 'monkeydata_training.mat' is in your path)
load('monkeydata_training.mat');  % loads the variable 'trial'

% --- Select a trial, reaching angle, and neural unit ---
trial_idx = 1;
angle_idx = 1;
unit_idx  = 1;

% Extract the spike train and hand position for the selected trial
spikeTrain = trial(trial_idx, angle_idx).spikes(unit_idx, :);
handPos    = trial(trial_idx, angle_idx).handPos;  % 3 x T matrix
% For movement, we choose the horizontal position (first row)
handX = handPos(1,:);
handY = handPos(2,:);

% --- Smooth the Spike Train ---
% Since the spike train is binary and sparse, we smooth it using a Gaussian kernel.
windowSize = 20; % window length in ms
sigma = 3;       % standard deviation for the Gaussian kernel
gaussKernel = fspecial('gaussian', [1, windowSize], sigma);
smoothedSpikeTrain = conv(double(spikeTrain), gaussKernel, 'same');

% --- Compute the STFT ---
fs = 1000;           % Sampling frequency (1 ms bins)
window_length = 100; % Window length in samples (e.g., 100 ms)
noverlap = 50;       % Overlap between windows (50 ms)
nfft = 256;          % Number of FFT points
[s, f, t] = spectrogram(smoothedSpikeTrain, window_length, noverlap, nfft, fs);

% --- Plot the STFT ---
figure;
% Create the primary axis for the spectrogram
ax1 = axes;
imagesc(t, f, abs(s));
axis xy;  % Correct orientation: time on x-axis, frequency on y-axis
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('STFT with Hand Movement Overlay');
colorbar;
hold on;

% --- Overlay Hand Movement ---
% Create an overlay axis that shares the same position but with a transparent background.
ax2 = axes('Position', get(ax1, 'Position'), 'Color', 'none');
hold(ax2, 'on');

% Compute time vector for the trial (hand position is sampled at 1000 Hz)
time_trial = (0:length(handX)-1) / fs;

% Plot the hand position (horizontal movement) in red
plot(ax2, time_trial, handY, 'r', 'LineWidth', 2);
ylabel(ax2, 'Hand Position (mm)', 'Color', 'r');

% Remove duplicate x-axis labels from the overlay and link the time axes
set(ax2, 'XAxisLocation', 'top', 'Color', 'none', 'XColor', 'k', 'YColor', 'r');
linkaxes([ax1, ax2], 'x');
ax2.XAxis.Visible = 'off';  % Optionally hide the top x-axis

% Optional: Adjust ax2 y-limits for clarity (e.g., padding around hand position data)
ax2.YLim = [min(handX)-10, max(handX)+10];
