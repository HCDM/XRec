import os
import json
import config

SRC_FILE = config.SRC_FILE
DATASET = config.DATASET


def _load_ratebeer():
    with open(SRC_FILE, encoding='ISO-8859-1') as f:
        lines = []
        for line in f:
            line = line.strip()
            if not line and lines:
                try:
                    entity = dict(
                        iid=lines[1].replace('beer/beerId: ', '').strip(),
                        uid=lines[-2].replace('review/profileName: ', '').strip(),
                        rating=int(
                            lines[-4].replace('review/overall: ', '').split('/')[0]),
                        text=lines[-1].replace('review/text: ', '').lower().strip(),
                        cat=lines[4].replace('beer/style: ', '').lower().split(' - ')[0].strip()
                    )
                    yield entity
                except Exception as e:
                    print('Read data error')
                    print(lines)
                    raise e

                lines = []
            else:
                lines.append(line)


def _load_tripadvisor():
    aspects = {'service', 'rooms', 'overall', 'value', 'location', 'cleanliness'}

    for fname in os.listdir(SRC_FILE):
        if not fname.endswith('.json'):
            continue

        iid = fname.split('.')[0].strip()

        with open(os.path.join(SRC_FILE, fname)) as f:
            obj = json.load(f)

            for review in obj['Reviews']:
                try:
                    if 'Ratings' in review:
                        ratings = review['Ratings']
                    else:
                        ratings = review['Overall']

                    # {'Service': '5', 'Cleanliness': '5', 'Overall': '5.0', 'Value': '5', 'Sleep Quality': '5', 'Rooms': '5', 'Location': '5'}
                    ratings = {
                        k.lower(): int(float(v)) for k, v in ratings.items()
                        if k.lower() in aspects
                    }

                    entity = dict(
                        iid=iid,
                        uid=review['Author'],
                        rating=ratings,
                        # rating=int(float(ratings['overall'])),
                        text=review['Content'].lower(),
                        cat=''
                    )

                    yield entity

                except Exception:
                    print('parse data error')
                    print(review)


def load_src():
    return dict(
        rb=_load_ratebeer,
        ta=_load_tripadvisor
    )[DATASET]()
