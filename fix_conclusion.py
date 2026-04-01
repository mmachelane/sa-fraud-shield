import json

with open("notebooks/02_sim_swap_model.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

diagram = """
```
Transaction --> Feature Builder (34 features) --> LightGBM
                                                      |
                                             SIM swap score (0-1)
                                                      |
                         +---------- GNN proxy x0.85 ----------+
                         |                                      |
                    Ensemble: 0.6 x SIM swap + 0.4 x GNN
                         |
               APPROVE / STEP_UP / BLOCK
                         |
          SHAP explanation + LLM narrative
                 (English + isiZulu)
```
"""

src = "".join(nb["cells"][33]["source"])
# Remove trailing blank lines then append diagram
src = src.rstrip() + "\n" + diagram

nb["cells"][33]["source"] = [src]

with open("notebooks/02_sim_swap_model.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("done")
print(repr(src[-300:]))
