from src.features import sql_queries
import argparse

parser = argparse.ArgumentParser(description='Download database')
parser.add_argument('--host', dest='host', default='localhost',
                   help='host')
parser.add_argument('--port', dest='port', type=int, default=3306,
                   help='port')
parser.add_argument('--user', dest='user', default='root',
                   help='user')
parser.add_argument('--password', dest='password', default='',
                   help='password')
parser.add_argument('--db', dest='db', default='flex',
                   help='db')


def db_data(host, port, user, password, db):
    args = parser.parse_args()


    dbc = {'host': host,
           'user': user,
           'password': password,
           'db': db,
           'charset': 'utf8mb4',
           'port': port
           }

    db_flex = sql_queries.Sql(dbc)

    sesiones = db_flex.pandas_query("""select * from sesiones""")

    perfiles_usuario = db_flex.pandas_query("""select * from perfiles_usuario""")

    movimientos = db_flex.pandas_query("""select * from movimientos""")

    db_flex.save_query(sesiones, 'sesiones')

    db_flex.save_query(perfiles_usuario, 'perfiles_usuario')

    db_flex.save_query(movimientos, 'movimientos')


if __name__ == '__main__':
    args = parser.parse_args()
    db_data(**args.__dict__)

