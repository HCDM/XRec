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
            rating=int(lines[-4].replace('review/overall: ', '').split('/')[0]),
            text=lines[-1].replace('review/text: ', '').lower().strip(),
          )
          yield entity
        except Exception as e:
          print('Read data error')
          print(lines)
          raise e

        lines = []
      else:
        lines.append(line)


def _load_yelp():
  with open(SRC_FILE, encoding='utf-8') as f:
    for line in f:
      record = json.loads(line)
      entity = dict(
        iid=record['business_id'],
        uid=record['user_id'],
        rating=record['stars'],
        text=record['text']
      )
      yield entity

def load_src():
  return dict(
    yelp=_load_yelp,
    ratebeer=_load_ratebeer
  )[DATASET]()
