from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from energy_efficiency.constant.constant import CURRENT_TIMESTAMP, DUMP_DATA_LINE
from energy_efficiency.util.util import *
from energy_efficiency.logger import logging
from energy_efficiency.exception import energy_exception
from energy_efficiency.constant import *
import os,sys
class DataCollection:
    def __init__(self):
        self.timestamp=CURRENT_TIMESTAMP
    def get_data_from_database(self):
        logging.info("Inside get data from dtabase")
        """
        This class will collect the data from the cassnadra database and this will take the data
        """
        cloud_config= {
        'secure_connect_bundle': 'E:\ml project\secure-connect-energy-efficient.zip'}
        auth_provider = PlainTextAuthProvider('qfXJZoknpszuzBCIgqRjWIwf', 'SZWI7H7DbzhitWmNPwBmFFWeRxF8WO8PA+O0dfbECQ9c0SbuR7TQCKqyylbHsL1LfOQSw4ISe-MKruHy75_knGCQiRqx_UXKsZbFx0zCoqpcm--9DU+Uz+jL1LI0d6lb')
        logging.info("connection established sucessfully")
        cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
        session = cluster.connect("energy_keyspace")
        logging.info("connected with keyspace")
        session.row_factory = pandas_factory
        session.default_fetch_size = None
        logging.info("keyspace connection done")
        query = "SELECT * from energy_keyspace.energy"
        rslt = session.execute(query, timeout=None)
        df = rslt._current_rows
        logging.info(f"df created {df}")
        return df
    def dumping_data(self):
        """
        This class will dump the data as a csv file in given path
        """
        try:
            df=self.get_data_from_database()

            logging.info(f"Getting data from database {df}")
            os.makedirs(os.path.join(DUMP_DATA_LINE,self.timestamp),exist_ok=True)
            path=os.path.join(os.path.join(DUMP_DATA_LINE,self.timestamp),"dumped_data.csv")
            df.to_csv(path,index=False)
            return path
        except Exception as e:
            raise energy_exception(e,sys) from e

