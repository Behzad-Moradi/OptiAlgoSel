import sqlite3


def create_request_table(db_dir):
    conn = sqlite3.connect(db_dir)
    cur = conn.cursor()
    
    cur.execute('''CREATE TABLE IF NOT EXISTS requests(
        request_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT NOT NULL,
        request_time DATETIME NOT NULL,
        request_status TEXT NOT NULL, -- processing, completed, failed
        problem_name TEXT NOT NULL,
        problem_dim INTEGER NOT NULL,
        num_sample_points INTEGER NOT NULL
        )''')
    
    conn.commit()
    cur.close()
    conn.close()
    
    return

################################################################################################

def create_result_table(db_dir):
    conn = sqlite3.connect(db_dir)
    cur = conn.cursor()
    
    cur.execute('''CREATE TABLE IF NOT EXISTS results(
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id INTEGER NOT NULL,
        predicted_algorithms TEXT NOT NULL,
        prediction_time DATETIME NOT NULL,
        FOREIGN KEY(request_id) REFERENCES requests(request_id)
        )''')
    
    conn.commit()
    cur.close()
    conn.close()
    
    return

################################################################################################
################################################################################################

def create_database_schema(db_dir):
        
    create_request_table(db_dir)
    create_result_table(db_dir)
    
    return