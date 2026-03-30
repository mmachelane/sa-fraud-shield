import math

import pandas as pd

from models.sim_swap.model import SimSwapDetector

d = SimSwapDetector.load("models/sim_swap/artifacts")
assert d._booster is not None
print("trees:", d._booster.num_trees())
features = {col: 0.0 for col in d.feature_names}
features["time_since_sim_swap_minutes"] = 30.0
features["new_device_first_tx"] = 1.0
features["amount_zar"] = 15000.0
features["log_amount"] = math.log1p(15000)
features["payment_rail_PAYSHAP"] = 1.0
features["merchant_category_peer_transfer"] = 1.0
df = pd.DataFrame([features])[d.feature_names]
print("Score:", d.predict_proba(df)[0])
