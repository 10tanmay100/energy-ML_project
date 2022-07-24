from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from energy_efficiency.constant.constant import DUMP_DATA_LINE
from energy_efficiency.util.util import *
from energy_efficiency.logger import logging
from energy_efficiency.exception import energy_exception
from energy_efficiency.constant.constant import *
from energy_efficiency.component.get_data import DataCollection

class DataValidation:
    def __init__(self):
        data=DataCollection()
        data.get_data_from_database()
        self.path=pd.read_csv(data.dumping_data())
        self.timestamp=CURRENT_TIMESTAMP
    def validate_data(self):
        logging.info("Validating data")
        l=[]
        for cols in self.path.columns:
            if cols in list(read_yaml_file(SCHEMA_FILE_PATH).keys()):
                if read_yaml_file(SCHEMA_FILE_PATH)[cols]==str(self.path[cols].dtypes):
                    l.append(True)
        if len(l)==len(list(read_yaml_file(SCHEMA_FILE_PATH).keys())):
            df=pd.DataFrame(self.path)
            os.makedirs(os.path.join(VALIDATE_PATH,self.timestamp),exist_ok=True)
            path_val=os.path.join(VALIDATE_PATH,self.timestamp,"validate.csv")
            df.to_csv(path_val,index=False)
            logging.info("Validation Successfull")
        else:
            logging.info("Validated failed")
        return path_val


        







