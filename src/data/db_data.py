from src.features import sql_queries

dbc = {'host': 'localhost',
       'user': 'root',
       'password': '',
       'db': 'flex',
       'charset': 'utf8mb4',
       'port': 3306
       }

db_flex = sql_queries.Sql(dbc)

sesiones = db_flex.pandas_query("""select * from sesiones""")

perfiles_usuario = db_flex.pandas_query("""select * from perfiles_usuario""")

db_flex.save_query(sesiones, 'sesiones')

db_flex.save_query(perfiles_usuario, 'perfiles_usuario')



