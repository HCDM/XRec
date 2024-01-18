''' get categories '''

from utils import load_src
from collections import Counter
import json
import config


def main():
    visited = set()
    cats = Counter()
    i_cats = Counter()

    with open(config.ITEM_FILE) as f:
        iids = f.read().split('\n')
        item_map = {iid: i for i, iid in enumerate(iids)}
        i_idices = {i_idx: None for i_idx in item_map.values()}

    for entity in load_src():
        if entity['iid'] in visited:
            continue

        visited.add(entity['iid'])
        cats[entity['cat']] += 1

        i_idx = item_map[entity['iid']] if entity['iid'] in item_map else None
        if i_idx is not None:
            i_idices[i_idx] = entity['cat']
            i_cats[entity['cat']] += 1

    cats = sorted([*cats.items()], key=lambda t: t[1])
    i_cats = sorted([*i_cats.items()], key=lambda t: t[1])
    for cat, n in cats:
        print(cat, n)
    print()
    for cat, n in i_cats:
        print(cat, n)

    cids = {cat: i for i, (cat, n) in enumerate(i_cats)}

    with open(config.ITEM_CATS_FILE, 'w+') as f:
        json.dump({iid: cids[cat] for iid, cat in i_idices.items()}, f)


if __name__ == '__main__':
    main()
