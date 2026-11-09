% FFT motion-removal core in plain MATLAB syntax (no toolboxes; also runs in GNU Octave).
% Input : results/matlab_windows.csv, 20 rows = [ppg(256) accx(256) accy(256) accz(256)], already band-passed 0.5-4 Hz
% Output: results/matlab_peaks.csv (window_row, plain_bpm, removed_bpm, motion_weight)
%         data/matlab_clean_spectra.csv (cleaned PPG spectra, compared against Python)
fs = 32; nfft = 1024; L = 256; motion_g = 0.05; lam = 1.0;
here = fileparts(mfilename('fullpath')); root = fileparts(here); addpath(here);
X = csvread(fullfile(root, 'results', 'matlab_windows.csv'));
n = size(X, 1);
f = (0:nfft/2)' * fs / nfft;
w = 0.5 - 0.5 * cos(2 * pi * (0:L-1)' / (L - 1));   % symmetric Hann = numpy.hanning
out = zeros(n, 4); Sc_all = zeros(n, nfft/2 + 1);
for i = 1:n
  ppg = X(i, 1:L)';
  acc = reshape(X(i, L+1:end), L, 3);
  Sp = nspec(ppg, w, nfft);
  Sm = max([nspec(acc(:,1), w, nfft), nspec(acc(:,2), w, nfft), nspec(acc(:,3), w, nfft)], [], 2);
  wt = min(max(max(std(acc, 1)) / motion_g, 0), 1);
  Sc = max(Sp - lam * wt * Sm, 0);
  out(i, :) = [i, peak_bpm(Sp, f, fs, nfft), peak_bpm(Sc, f, fs, nfft), wt];
  Sc_all(i, :) = Sc';
end
fid = fopen(fullfile(root, 'results', 'matlab_peaks.csv'), 'w');
fprintf(fid, 'row,plain_bpm,removed_bpm,motion_weight\n');
fprintf(fid, '%d,%.10f,%.10f,%.10f\n', out');
fclose(fid);
dlmwrite(fullfile(root, 'data', 'matlab_clean_spectra.csv'), Sc_all, 'precision', '%.12e');
disp(out);
