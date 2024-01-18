# Data Preprocessing

- Extract feature set if not given `python extract_features.py`

- Recursively filter user and item candidates with enough reviews towards each others `python filter_user_item_lists.py`

- Extract data with explanations from raw data in respect to filtered users and items `python extact_exp_data.py -w=4`

- Generate vocabulary `python gen_voc.py`

- OPTIONAL: Remove data with too many unknown tokens `python purge_data.py`

- Split data into train, val and test `python split_data.py`

- Download & Filter word embeddings `python get_word_embedding.py`
