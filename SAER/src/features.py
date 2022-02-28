import json

import config


with open(config.FEATURE_FILE) as f:
    features = set(json.load(f))

with open(config.ITEM_FEATURE_FILE) as f:
    item_features = {int(i): f for i, f in json.load(f).items()}
