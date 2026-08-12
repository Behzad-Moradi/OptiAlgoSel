from Utils.get_con_db import get_connection

def create_request_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('''CREATE TABLE IF NOT EXISTS requests(
                request_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY, 
                user_email TEXT NOT NULL,
                request_time TIMESTAMPTZ NOT NULL,
                request_status TEXT NOT NULL, -- processing, completed, failed
                problem_name TEXT NOT NULL,
                problem_dim INTEGER NOT NULL,
                num_sample_points INTEGER NOT NULL
                )''')
    
    return

################################################################################################

def create_result_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('''CREATE TABLE IF NOT EXISTS results(
                result_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                request_id INTEGER NOT NULL,
                predicted_algorithms TEXT NOT NULL,
                prediction_time TIMESTAMPTZ NOT NULL,
                FOREIGN KEY(request_id) REFERENCES requests(request_id)
                )''')
 
    return

################################################################################################
################################################################################################

def create_database_schema_pg():
    
    create_request_table()
    create_result_table()
    
    print("Database schema created successfully.")
    return