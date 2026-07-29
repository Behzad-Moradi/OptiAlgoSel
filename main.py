from Utils.create_database import create_tables, init_tables
import sqlite3

def main():
    
    create_tables()
    init_tables()
   
    return

if __name__ == "__main__":
    main()