% Peak in 40-210 bpm with parabolic interpolation (same rule as src/spectral.py peak_bpm).
function bpm = peak_bpm(S, f, fs, nfft)
  idx = find(f * 60 >= 0.67 * 60 & f * 60 <= 3.5 * 60);
  [~, j] = max(S(idx)); k = idx(j);
  a = S(k-1); b = S(k); c = S(k+1); den = a - 2*b + c;
  if den ~= 0, off = 0.5 * (a - c) / den; else, off = 0; end
  off = min(max(off, -0.5), 0.5);
  bpm = ((k - 1) + off) * fs / nfft * 60;   % k is 1-based
end
