from logging import RootLogger
from energy_efficiency.constant.constant import *
from energy_efficiency.util.util import *
from energy_efficiency.logger import logging
from energy_efficiency.exception import energy_exception
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import pickle
import shutil
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from energy_efficiency.constant.constant import *
from energy_efficiency.component.get_data import DataCollection
from energy_efficiency.component.do_validation import DataValidation
from collections import namedtuple
from energy_efficiency.component.data_transform import DataTransformation
from energy_efficiency.component.model_training import model_training

class pusher:
    def __init__(self):
        m=model_training()
        d=DataTransformation()
        path_dir,path_pkl=d.transform_data()
        self.pre=path_pkl
        model1,score1,score2,model2=m.model_trainer()
        self.model1=model1[score1.index(max(score1))]
        self.model2=model2[score2.index(max(score2))]
    def deploy(self):
        f=open(os.path.join(ROOT_DIR,"model_cool.pkl"),"wb")
        pickle.dump(self.model1,f)
        f1=open(os.path.join(ROOT_DIR,"model_heat.pkl"),"wb")
        pickle.dump(self.model2,f)
        shutil.copy(self.pre,ROOT_DIR)
        print('done push')
