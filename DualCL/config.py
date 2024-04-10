import mysql.connector

def connect_to_mysql(host, port, database, user, password):
    try:
        connection = mysql.connector.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        return connection
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL database: {err}")
        return None

def execute_query(connection, query):
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query)
        records = cursor.fetchall()
        return records
    except mysql.connector.Error as err:
        print(f"Error executing query: {err}")
        return []
