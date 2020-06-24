import pymysql
import pandas as pd

# Connect to the database.
con = pymysql.connect(host='localhost',
                      user='root',
                      password='',
                      db='flex',
                      charset='utf8mb4',
                      port=3306,
                      cursorclass=pymysql.cursors.DictCursor)

# prepare a cursor object using cursor() method
cursor = con.cursor()

# ejecuta el SQL query usando el metodo execute().
cursor.execute("SELECT VERSION()")

try:
    with con.cursor() as cursor:
        # Read all records
        sql = """select * from perfiles_usuario 
       """
        cursor.execute(sql)
        result = cursor.fetchall()
finally:
    pass

# Parse dict to pandas
df = pd.DataFrame.from_dict(result)

# Save data raw
df.to_parquet('data/raw/flex_perfiles_usuario.parquet')