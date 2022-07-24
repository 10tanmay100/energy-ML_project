from energy_efficiency.constant.constant import DUMP_DATA_LINE
from energy_efficiency.util.util import *
from energy_efficiency.logger import logging
from energy_efficiency.exception import energy_exception
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from energy_efficiency.constant.constant import *
from energy_efficiency.component.get_data import DataCollection
from energy_efficiency.component.do_validation import DataValidation
from collections import namedtuple

class DataTransformation:
    def __init__(self):
        validate=DataValidation()
        validated_path=validate.validate_data()
        self.timestamp=CURRENT_TIMESTAMP
        self.path=pd.read_csv(validated_path)
    def transform_data(self):
        try:
            logging.info("Inside transform data")
            num_pipeline = Pipeline(steps=[
                ('imputer', KNNImputer(n_neighbors=2)),
                ('scaler', StandardScaler())])
            cols_transform=["glazing_area","glazing_area_distribution","orientation",'overall_height',"relative_compactness","roof_area","surface_area","wall_area"]
            logging.info("pipeline created")
            preprocessing = ColumnTransformer([('num_pipeline', num_pipeline,cols_transform)])
            os.makedirs(os.path.join(TRANSFORMED_PATH,self.timestamp),exist_ok=True)
            logging.info("preprocessing file craeted")
            path_dir=os.path.join(TRANSFORMED_PATH,self.timestamp,"transformed.csv")
            df=self.path[["glazing_area","glazing_area_distribution","orientation",'overall_height',"relative_compactness","roof_area","surface_area","wall_area"]]
            df1=pd.DataFrame(preprocessing.fit_transform(df),columns=df.columns)
            df1=df1.reset_index(drop=True)
            df2=self.path[["cooling_load","heating_load"]].reset_index(drop=True)
            pd.concat([df1,df2],axis=1).to_csv(path_dir,index=False)
            os.makedirs(os.path.join(TRANSFORMED_PATH_PICKLE,self.timestamp))
            path_pkl=os.path.join(TRANSFORMED_PATH_PICKLE,self.timestamp,"preprocessed.pkl")
            with open(path_pkl,"wb") as f:
                pickle.dump(preprocessing,f)
            logging.info("Pickle file created")
            tuples=namedtuple("tuples",["transformed_path","pickle_transform_path"])
            return tuples(transformed_path=path_dir,pickle_transform_path=path_pkl)
        except Exception as e:
            raise energy_exception(e,sys) from e
        





