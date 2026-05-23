import os
import tarfile
import sqlite3

from dotenv import load_dotenv
from tqdm.auto import tqdm
import pandas as pd


load_dotenv()
FOOD_TAR_PATH = os.environ.get('FOOD_TAR_PATH')
DB = os.environ.get('DB', 'embedding.db')


def initial_setup(model='clip'):
    paths = []
    with tarfile.open(FOOD_TAR_PATH, 'r:gz') as tar:
        for member in tqdm(tar.getmembers()):
            if member.isfile():
                paths.append(member.name)

    df = pd.DataFrame(paths, columns=['path'])
    df = df[df['path'].str.endswith('.jpg')]
    df['id'] = df['path'].str.split('/').map(lambda x: int(x[-1].split('.')[0]))
    df['label'] = df['path'].str.split('/').map(lambda x: x[-2])
    df['model'] = model

    with sqlite3.connect(DB) as conn:
        sql = 'INSERT OR IGNORE INTO embeddings (id, model, path, label) VALUES (?, ?, ?, ?)'
        conn.executemany(sql, df[['id', 'model', 'path', 'label']].values)

    return df