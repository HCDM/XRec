# Data Preprocessing

- Extract feature set if not given `python extract_features.py`

- Recursively filter user and item candidates with enough reviews towards each others `python filter_user_item_lists.py`

- Extract data with explanations from raw data with respect to filtered users and items `python extact_exp_data.py -w=4`

- Split data into train, val and test `python split_data.py`

- Prepare a fixed NDCG list `python prep_ndcg_data.py`

- Extract item features `python extract_item_features.py`

- Generate vocabulary `python gen_voc.py`

- Download & Filter word embeddings `python get_word_embedding.py`
