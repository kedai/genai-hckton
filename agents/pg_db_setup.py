import psycopg2
import json
import glob
from datetime import datetime

def main():
    # Database connection parameters
    db_params = {
        'dbname': 'exchange_rates',
        'user': 'root',
        'password': 'postgres',
        'host': 'localhost',
        'port': '5432',
    }

    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(**db_params)
        print("Connected to the database successfully.")
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return

    cur = conn.cursor()

    # Create table if not exists
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS exchange_rates (
        id SERIAL PRIMARY KEY,
        currency_code VARCHAR(10),
        unit INTEGER,
        date DATE,
        buying_rate NUMERIC,
        selling_rate NUMERIC,
        middle_rate NUMERIC,
        quote VARCHAR(10),
        session VARCHAR(10),
        last_updated TIMESTAMP
    );
    '''
    try:
        cur.execute(create_table_query)
        conn.commit()
        print("Table 'rates' is ready.")
    except Exception as e:
        print(f"Error creating table: {e}")
        conn.rollback()
        return

    # Path to your JSON files
    json_files = glob.glob('exchange_rates/*.json')

    for file in json_files:
        print(f"Processing file: {file}")
        with open(file, 'r') as f:
            data = json.load(f)

        # Extract data
        try:
            currency_code = data['data'].get('currency_code')
            unit = data['data'].get('unit')
            rates = data['data'].get('rate', [])
            meta = data.get('meta', {})
            quote = meta.get('quote')
            session = meta.get('session')
            last_updated_str = meta.get('last_updated')
            if last_updated_str:
                last_updated = datetime.strptime(last_updated_str, '%Y-%m-%d %H:%M:%S')
            else:
                last_updated = None

            # Insert rates into the database
            for rate in rates:
                date_str = rate.get('date')
                if date_str:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                else:
                    date_obj = None

                buying_rate = rate.get('buying_rate')
                selling_rate = rate.get('selling_rate')
                middle_rate = rate.get('middle_rate')

                insert_query = '''
                INSERT INTO exchange_rates (currency_code, unit, date, buying_rate, selling_rate, middle_rate, quote, session, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                '''
                try:
                    cur.execute(insert_query, (
                        currency_code,
                        unit,
                        date_obj,
                        buying_rate,
                        selling_rate,
                        middle_rate,
                        quote,
                        session,
                        last_updated
                    ))
                    conn.commit()
                    print(f"Inserted rate for date {date_str}.")
                except Exception as e:
                    print(f"Error inserting data: {e}")
                    conn.rollback()
        except AttributeError: print('no data')

    # Close the connection
    cur.close()
    conn.close()
    print("Database connection closed.")

if __name__ == '__main__':
    main()
