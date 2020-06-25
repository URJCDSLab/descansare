import pymysql
import pandas as pd


class Sql:

    def __init__(self, dbc):
        """Connection to data base and
        execute SQL query using execute() method"""
        db = pymysql.connect(**dbc, cursorclass=pymysql.cursors.DictCursor)
        self.dbc = dbc
        self.cursor = db.cursor()
        self.cursor.execute("SELECT VERSION()")

    def query(self, sql):
        """Connect to the database"""
        try:
            self.cursor.execute(sql)
            return self.cursor.fetchall()
        finally:
            pass

    def pandas_query(self, sql):
        """Parse query result to pandas object"""
        result = self.query(sql)
        df = pd.DataFrame.from_dict(result)
        return df

    def save_query(self, df, name):
        """Save pandas as parquet"""
        db_name = self.dbc['db']
        df.to_parquet(f'data/raw/{db_name}_{name}.parquet')