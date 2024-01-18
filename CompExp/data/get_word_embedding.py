
'''
download the word embedding and filter it based on our corpus
'''

import urllib.request
import os
import shutil
import zipfile
import json

import config

DIR_PATH = os.path.dirname(__file__)

GLOVE_WE_URL = 'http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip'

WE_FOLDER = os.path.join(DIR_PATH, 'word_embedding')
ZIP_FILE = os.path.join(WE_FOLDER, 'glove.42B.300d.zip')
WE_FILE = os.path.join(WE_FOLDER, 'glove.42B.300d.txt')
EMBEDDING_SIZE = 300

VOC_FILE = config.VOC_FILE
VOC_WE_FILE = config.VOC_WE_FILE


def main():
    if not os.path.exists(WE_FOLDER):
        print('create folder', WE_FOLDER)
        os.mkdir(WE_FOLDER)

    if not os.path.exists(WE_FILE):
        print('downloading embedding from', GLOVE_WE_URL)
        # Download the file from `url` and save it locally under:
        with urllib.request.urlopen(GLOVE_WE_URL) as response, open(ZIP_FILE, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(WE_FOLDER)

        os.remove(ZIP_FILE)
        print('downloaded embedding from', GLOVE_WE_URL)

    with open(VOC_FILE) as vf:
        voc = json.load(vf)

    default_ebd = ' '.join(['0.0'] * EMBEDDING_SIZE)
    we = {t: default_ebd for t in voc}

    # unknown
    we['<unk>'] = default_ebd

    with open(WE_FILE, 'r', encoding='utf8') as file:
        # emdedding file is too large, read line by line
        size = 0
        i = 1
        for line in file:
            line = line.rstrip('\n')
            space_idx = line.find(' ')

            token = line[:space_idx]

            if token in we:
                size += 1
                we[token] = line[space_idx+1:]

            if i % 50000 == 0:
                print('read word embedding lines:', i)

            i += 1

        print('size of the mapped embedding:', size)

    lines = [k + ' ' + we[k] for k in voc]

    with open(VOC_WE_FILE, 'w', encoding='utf8') as file:
        file.write('\n'.join(lines))


if __name__ == '__main__':
    main()
