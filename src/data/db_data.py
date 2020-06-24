import pymysql
import pandas as pd

# Connect to the database.
con = pymysql.connect(host='localhost',
                      user='victor',
                      password='machine1$',
                      db='pharaoh',
                      charset='utf8mb4',
                      port=6666,
                      cursorclass=pymysql.cursors.DictCursor)

# prepare a cursor object using cursor() method
cursor = con.cursor()

# ejecuta el SQL query usando el metodo execute().
cursor.execute("SELECT VERSION()")

try:
    with con.cursor() as cursor:
        # Read all records
        sql = """select o.id as id, null as destino1, c.name as canal, 
        s.name as servicio from opportunities o left join opportunities_destinations 
        d on o.id = d.opportunity left join channels c on o.channel = c.id 
        left join services s on o.service=s.id where o.organization = 1 and 
        (d.isOrigin = 1 and not exists (select * from opportunities_destinations d2 
        where o.id = d2.opportunity and d2.isOrigin = 0)) union select o.id as id, 
        d.text as destino1, c.name as canal, s.name as servicio from opportunities o 
        left join opportunities_destinations d on o.id = d.opportunity 
        left join channels c on o.channel = c.id left join services s on o.service=s.id 
        where o.organization = 1 and (d.isOrigin = 0 or not exists 
        (select null from opportunities_destinations d2 where o.id = d2.opportunity)) ORDER BY `id` ASC"""
        cursor.execute(sql)
        result = cursor.fetchall()
finally:
    pass

# Parse dict to pandas
df = pd.DataFrame.from_dict(result)
# Set correct index
df.set_index('id', inplace=True)
# Replace null by 'selecciona'
df.canal[df.canal.isnull()] = 'selecciona'
# Save data to predict
df.to_parquet('data/external/db_data.parquet')