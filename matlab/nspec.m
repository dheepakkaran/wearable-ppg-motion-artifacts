% Normalised |FFT| of a mean-removed, Hann-windowed window (MATLAB/Octave).
function S = nspec(x, w, nfft)
  x = x - mean(x);
  F = abs(fft(x .* w, nfft));
  S = F(1:nfft/2 + 1);
  S = S / max(S);
end
