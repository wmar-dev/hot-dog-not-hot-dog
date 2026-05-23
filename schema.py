import sqlite3


DB = 'embedding.db'


def setup_db():
    with sqlite3.connect(DB) as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER NOT NULL,
            model TEXT NOT NULL,
            path TEXT,
            label TEXT,
            embedding BLOB,
            PRIMARY KEY (id, model)
        )
        ''')
        conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER NOT NULL,
            model TEXT NOT NULL,
            predicted_label TEXT,
            PRIMARY KEY (id, model)
        )
        ''')
