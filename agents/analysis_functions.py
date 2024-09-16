# analysis_functions.py
import psycopg2
import pandas as pd
from dateutil import parser
import matplotlib.pyplot as plt
import io
import base64

def get_db_connection():
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='exchange_rates',
        user='root',
        password='postgres'
    )
    return conn


def query_to_dataframe(cursor):
    """
    Converts the results of a psycopg2 query to a pandas DataFrame.

    Args:
    cursor (psycopg2.cursor): A psycopg2 cursor with an executed query.

    Returns:
    pd.DataFrame: DataFrame containing the query results.
    """
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    print(f'cols: {columns}')
    df = pd.DataFrame(rows, columns=columns)
    print(f'rows: {rows}')

    return df

def get_all_rates(date_str, currency_code=None):
    """
    Retrieves exchange rates for a given date and optional currency code.
    """
    conn = get_db_connection()
    try:
        date_obj = parser.parse(date_str)
    except ValueError:
        conn.close()
        return f"Invalid date format: {date_str}"
    date_formatted = date_obj.strftime('%Y-%m-%d')

    query = "SELECT * FROM exchange_rates WHERE date = %s"
    params = [date_formatted]

    if currency_code:
        query += " AND currency_code = %s"
        params.append(currency_code.upper())

    try:
        cur = conn.cursor()
        cur.execute(query, params)
        df = query_to_dataframe(cur)
        cur.close()
        print(f'df: {df.head()}')
    except Exception as e:
        conn.close()
        return f"Error executing query: {e}"
    finally:
        conn.close()

    if df.empty:
        return f"No exchange rate data found for {currency_code or 'all currencies'} on {date_formatted}."
    else:
        return df.to_dict(orient='records')

def plot_currency_trends(currency_code, start_date_str, end_date_str):
    """
    Plots the exchange rate trends for a specific currency between two dates.
    Returns a base64-encoded PNG image.
    """
    conn = get_db_connection()
    try:
        start_date = parser.parse(start_date_str)
        end_date = parser.parse(end_date_str)
    except ValueError:
        conn.close()
        return f"Invalid date format. Please use a valid date format."

    start_date_formatted = start_date.strftime('%Y-%m-%d')
    end_date_formatted = end_date.strftime('%Y-%m-%d')

    query = """
        SELECT date, selling_rate, buying_rate, currency_code FROM exchange_rates
        WHERE currency_code = %s AND date BETWEEN %s AND %s
        AND session='1700'
        ORDER BY date
    """
    params = [currency_code.upper().replace("'",""), start_date_formatted, end_date_formatted]
    print(f"""cur: {currency_code.upper().replace("'","")}""")
    try:
        cur = conn.cursor()
        rendered_query = cur.mogrify(query, params).decode('utf-8')
        print(f'rendered: {rendered_query}')
        cur.execute(query, params)
        df = query_to_dataframe(cur)
        cur.close()
        print(f'df: {df.head()}')
    except Exception as e:
        conn.close()
        return f"Error executing query: {e}"
    finally:
        conn.close()

    if df.empty:
        return f"No exchange rate data found for {currency_code.upper()} between {start_date_formatted} and {end_date_formatted}."
    else:
        # Convert 'date' column to datetime
        df['date'] = pd.to_datetime(df['date'])
        # Ensure 'middle_rate' is numeric
        df['buying_rate'] = pd.to_numeric(df['buying_rate'], errors='coerce')
        df['selling_rate'] = pd.to_numeric(df['selling_rate'], errors='coerce')
        print(f'df: {df.head()}')

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(df['date'], df['buying_rate'], marker='o')
        plt.plot(df['date'], df['selling_rate'], marker='x', label='Selling Rate', linestyle='--')
        plt.title(f'Exchange Rate Trends for {currency_code.upper()}')
        plt.xlabel('Date')
        plt.ylabel('Exchange Rate')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        # Encode the image in base64
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        return encoded_image
