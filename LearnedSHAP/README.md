# fast_shapley

### Case study on explained shapley vs learned shapley on ViTs and ResNets.

1. `pip install -r requirements.txt`
2. `make build`
3. `make run`

### Repo Structure

`/gradient_shap` → Gradient SHAP experiments, see `gradient_shap.ipynb`.

`/harsanyi_shap` → Harsanyi SHAP experiments, see `harsanyi_shap.ipynb`.

`/data` → builds dataloaders and stores datasets

`/models` → holds model files (`harsanyi.py`, `resnet_3d.py`, `vit.py`, etc.)

note: opted for GradientExplainer rather than DeepSHAP due to lack of GELU/LayerNorm/Identity compatibility