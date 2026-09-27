# Research: evaluation behind the reported numbers

These scripts produced every number and figure in the README and in our implementation paper. Their outputs are already saved in `results/`, so you only need to re-run them if you change something.

**Setup:** install `requirements-dev.txt` (Python 3.12). The scripts expect the original project folders under `research/proj/` (`proj/audio_emotion_recognition_project`, `proj/pet_emotion_without_venv`). That folder is git-ignored.

| Order | Script | What it does | Output |
|---|---|---|---|
| 1 | `eval_saved.py audio`, `eval_saved.py image` | Re-evaluates the originally saved checkpoints on their hold-out splits | `results/saved_*.json` |
| 2 | `profile_data.py` | Dataset statistics and the duplicate audit (difference hash) | `results/data_profile.json` |
| 3 | `image_features.py` | Frozen ImageNet features, 48×48 gray tensors, duplicate groups | feature cache |
| 4 | `holdout_image.py` | Frozen-backbone heads on the recovered 750-image hold-out split | `results/img/holdout_*.json` |
| 5 | `cv_image.py`, `cv_audio.py` | Repeated stratified (group) 5-fold cross-validation (about 2 h on CPU) | `results/*/cv_summary.json` |
| 6 | `probe_audio.py` | Metadata-only decision tree (the recording-source bias) | `results/audio/probe.json` |
| 7 | `parse_logs.py`, `bench.py`, `analysis.py` | Notebook training curves, CPU latency, calibration and confidence tiers | `results/latency.json`, `results/analysis.json` |
| 8 | `diagrams.py`, `figures_data.py`, `figures_results.py` | All figures | `figs/` |
