import json

with open("notebooks/02_sim_swap_model.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

nb["cells"][2]["source"] = [
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.ticker as mtick\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import shap\n",
    "from sklearn.metrics import (\n",
    "    ConfusionMatrixDisplay,\n",
    "    PrecisionRecallDisplay,\n",
    "    RocCurveDisplay,\n",
    "    average_precision_score,\n",
    "    confusion_matrix,\n",
    "    roc_auc_score,\n",
    ")\n",
    "\n",
    "import warnings\n",
    'warnings.filterwarnings("ignore")\n',
    "\n",
    "# Resolve project root\n",
    'ROOT = Path(r"C:/Users/kmosw/Documents/Projects/sa-fraud-shield")\n',
    "sys.path.insert(0, str(ROOT))\n",
    "\n",
    "# Ensure plot output dir exists\n",
    '(ROOT / "docs" / "assets").mkdir(parents=True, exist_ok=True)\n',
    "\n",
    'plt.style.use("dark_background")\n',
    'ACCENT = "#4da6ff"\n',
    'FRAUD_COLOR = "#ff4d4d"\n',
    'LEGIT_COLOR = "#4dff88"\n',
    'WARN_COLOR = "#ffd700"\n',
    "\n",
    'print(f"ROOT: {ROOT}")\n',
    'print("Setup complete")\n',
]

with open("notebooks/02_sim_swap_model.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("done")
