import sqlite3


DB = 'embedding.db'


def setup_db():
    with sqlite3.connect(DB) as conn:
        sql = '''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY,
            path TEXT,
            label TEXT,
            embedding BLOB
        )
        '''
        conn.execute(sql)
        sql = '''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            predicted_label TEXT
        )
        '''
        conn.execute(sql)        