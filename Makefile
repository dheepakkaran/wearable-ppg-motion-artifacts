# Reproduce everything from the public PPG-DaLiA download (UCI #495).
PY      = .venv/bin/python
# uv and pixi: taken from PATH; a local tools folder is used only if it exists
TOOLS := $(HOME)/workspace/.tools
ifneq ($(wildcard $(TOOLS)/bin/uv),)
UV ?= $(TOOLS)/bin/uv
PIXI ?= $(TOOLS)/bin/pixi
export UV_CACHE_DIR ?= $(TOOLS)/uv-cache
export UV_PYTHON_INSTALL_DIR ?= $(TOOLS)/python
export PIXI_CACHE_DIR ?= $(TOOLS)/pixi-cache
else
UV ?= uv
PIXI ?= pixi
endif

.PHONY: reproduce env data windows features evaluate test matlab clean-derived

reproduce: env data windows test features evaluate matlab

env:
	test -x $(PY) || $(UV) venv -p 3.12 .venv
	$(UV) pip install -q -p $(PY) -r requirements.txt --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu

data:
	$(PY) scripts/fetch.py

windows:            # one subject pickle in RAM at a time (lean streaming unpickler)
	$(PY) src/extract.py

test:
	$(PY) -m pytest -q tests

features:
	$(PY) -m src.build_features

evaluate:           # LOSO for every method -> results/metrics.json (CNN ~ a few min on CPU)
	$(PY) -m src.evaluate

matlab:             # Octave cross-check of the FFT motion-removal core (skip if Octave is unavailable)
	$(PY) scripts/matlab_export.py
	$(PIXI) run octave --no-gui -q --eval "run('matlab/spectral_check.m')" > results/octave_run.txt 2>&1 || echo "octave not available"
	$(PY) scripts/matlab_compare.py

clean-derived:
	rm -rf data/windows data/features.csv.gz data/cnn_inputs.npz
