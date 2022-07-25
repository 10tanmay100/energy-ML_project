from gc import get_debug
from tkinter import Grid
from energy_efficiency.constant.constant import *
from energy_efficiency.util.util import *
from energy_efficiency.logger import logging
from energy_efficiency.exception import energy_exception
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import pickle
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from energy_efficiency.constant.constant import *
from energy_efficiency.component.get_data import DataCollection
from energy_efficiency.component.do_validation import DataValidation
from collections import namedtuple
from energy_efficiency.component.data_transform import DataTransformation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso,LassoCV,Ridge,RidgeCV
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

class model_training:
    def __init__(self):
        transform=DataTransformation()
        path_dir,path_pkl=transform.transform_data()
        self.path=pd.read_csv(path_dir)
        self.timestamp=CURRENT_TIMESTAMP
    def splitting_the_data(self):
        x1=self.path.drop(["cooling_load","heating_load"],axis=1)
        y1=self.path[["cooling_load"]]
        x1=x1[["glazing_area","relative_compactness","roof_area","wall_area"]]
        x_train1,x_test1,y_train1,y_test1=train_test_split(x1,y1,test_size=0.3,random_state=0)
        cool_train=pd.concat([x_train1,y_train1],axis=1)
        os.makedirs(os.path.join(COOLING_TRAIN,self.timestamp),exist_ok=True)
        cool_train.to_csv(os.path.join(COOLING_TRAIN,self.timestamp,"cool_train.csv"))
        cool_test=pd.concat([x_test1,y_test1],axis=1)
        os.makedirs(os.path.join(COOLING_TEST,self.timestamp),exist_ok=True)
        cool_test.to_csv(os.path.join(COOLING_TEST,self.timestamp,"cool_test.csv"))
        
        x2=self.path.drop(["cooling_load","heating_load"],axis=1)
        y2=self.path[["heating_load"]]
        x2=x2[["glazing_area","relative_compactness","roof_area","wall_area"]]
        x_train2,x_test2,y_train2,y_test2=train_test_split(x2,y2,test_size=0.3,random_state=0)
        heat_train=pd.concat([x_train2,y_train2],axis=1)
        os.makedirs(os.path.join(HEATING_TRAIN,self.timestamp),exist_ok=True)
        heat_train.to_csv(os.path.join(HEATING_TRAIN,self.timestamp,"heat_train.csv"))
        heat_test=pd.concat([x_test2,y_test2],axis=1)
        os.makedirs(os.path.join(HEATING_TEST,self.timestamp),exist_ok=True)
        heat_test.to_csv(os.path.join(HEATING_TEST,self.timestamp,"heat_test.csv"))
        return x_train1,x_test1,y_test1,y_train1,x_train2,x_test2,y_test2,y_train2
    def model_trainer(self,base_score=0.6):
        try:
            x_train1,x_test1,y_test1,y_train1,x_train2,x_test2,y_test2,y_train2=self.splitting_the_data()
            logging.info("Getting splitting data")
            score1=[]
            model1=[]
            lr=LinearRegression()
            logging.info(f"using {lr}")
            lr_params={"fit_intercept":[True,False]}
            gr=GridSearchCV(estimator=lr,param_grid=lr_params)
            gr.fit(x_train1,y_train1)
            fit_intercept_gr=gr.best_params_
            logging.info(f"best params {fit_intercept_gr} for {lr}")
            lr=LinearRegression(fit_intercept=fit_intercept_gr)
            lr.fit(x_train1,y_train1)
            pred=lr.predict(x_test1)
            if r2_score(y_test1,pred)>=base_score:
                logging.info(f"{lr} accuracy score for cool is {r2_score(y_test1,pred)}")
                score1.append(r2_score(y_test1,pred))
                model1.append(lr)
            dr=DecisionTreeRegressor()
            logging.info(f"using {dr}")
            dr_params={"criterion":['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],"splitter":["best", "random"],
            "min_samples_split":[1,2,3,4,5,6,7,8],"max_features":['auto', 'sqrt', 'log2']}
            gr=GridSearchCV(estimator=dr,param_grid=dr_params)
            gr.fit(x_train1,y_train1)
            dr_gr=gr.best_params_
            logging.info(f"best params {dr_gr} for {gr}")
            dr=DecisionTreeRegressor(criterion=dr_gr["criterion"],splitter=dr_gr["splitter"],min_samples_split=dr_gr["min_samples_split"],max_features=dr_gr["max_features"])
            dr.fit(x_train1,y_train1)
            pred=dr.predict(x_test1)
            if r2_score(y_test1,pred)>=base_score:
                logging.info(f"{gr} accuracy score for cool is {r2_score(y_test1,pred)}")
                score1.append(r2_score(y_test1,pred))
                model1.append(dr)
            svr=SVR()
            logging.info(f"using {svr}")
            svr_params={"kernel":['linear', 'poly', 'rbf', 'sigmoid'],"gamma":['scale', 'auto']}
            gr=GridSearchCV(estimator=svr,param_grid=svr_params)
            gr.fit(x_train1,y_train1)
            svr_gr=gr.best_params_
            logging.info(f"best params {svr_gr} for {svr}")
            svr=SVR(kernel=svr_gr["kernel"],gamma=svr_gr["gamma"])
            svr.fit(x_train1,y_train1)
            pred=svr.predict(x_test1)
            if r2_score(y_test1,pred)>=base_score:
                logging.info(f"{svr} accuracy score for cool is {r2_score(y_test1,pred)}")
                score1.append(r2_score(y_test1,pred))
                model1.append(svr)
            rr=RandomForestRegressor()
            logging.info(f"using {rr}")
            rr_params={"criterion":['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],"min_samples_split":[1,2,3,4,5,6,7,8],"max_features":['sqrt', 'log2']}
            gr=GridSearchCV(estimator=rr,param_grid=rr_params)
            gr.fit(x_train1,y_train1)
            rr_gr=gr.best_params_
            logging.info(f"best params {rr_gr} for {rr}")
            rr=RandomForestRegressor(criterion=rr_gr["criterion"],min_samples_split=rr_gr["min_samples_split"],max_features=rr_gr["max_features"])
            rr.fit(x_train1,y_train1)
            pred=rr.predict(x_test1)
            if r2_score(y_test1,pred)>=base_score:
                logging.info(f"{rr} accuracy score for cool is {r2_score(y_test1,pred)}")
                score1.append(r2_score(y_test1,pred))
                model1.append(rr)
            gd=GradientBoostingRegressor()
            logging.info(f"using {gd}")
            gd_params={"loss":['squared_error', 'absolute_error', 'huber', 'quantile'],"learning_rate":[0.1,0.001,0.00001],"criterion":['friedman_mse', 'squared_error', 'mse'],"min_samples_split":[1,2,3,4,5,6,7,8],"max_features":['auto', 'sqrt', 'log2']}
            gr=GridSearchCV(estimator=gd,param_grid=gd_params)
            gr.fit(x_train1,y_train1)
            gd_gr=gr.best_params_
            logging.info(f"best params {gd_gr} for {gr}")
            gd=GradientBoostingRegressor(loss=gd_gr["loss"],learning_rate=gd_gr["learning_rate"],criterion=gd_gr["criterion"],min_samples_split=gd_gr["min_samples_split"],max_features=gd_gr["max_features"])
            gd.fit(x_train1,y_train1)
            pred=gd.predict(x_test1)
            if r2_score(y_test1,pred)>=base_score:
                logging.info(f"{gd} accuracy score for cool is {r2_score(y_test1,pred)}")
                score1.append(r2_score(y_test1,pred))
                model1.append(gd)
            logging.info("Heating score collection of models")
            score2=[]
            model2=[]
            gd=GradientBoostingRegressor()
            logging.info(f"using {gd}")
            gd_params={"loss":['squared_error', 'absolute_error', 'huber', 'quantile'],"learning_rate":[0.1,0.001,0.00001],"criterion":['friedman_mse', 'squared_error', 'mse'],"min_samples_split":[1,2,3,4,5,6,7,8],"max_features":['auto', 'sqrt', 'log2']}
            gr=GridSearchCV(estimator=gd,param_grid=gd_params)
            gr.fit(x_train2,y_train2)
            gd_gr=gr.best_params_
            logging.info(f"best params {gd_gr} for {gr}")
            gd=GradientBoostingRegressor(loss=gd_gr["loss"],learning_rate=gd_gr["learning_rate"],criterion=gd_gr["criterion"],min_samples_split=gd_gr["min_samples_split"],max_features=gd_gr["max_features"])
            
            gd.fit(x_train2,y_train2)
            pred=gd.predict(x_test2)
            if r2_score(y_test2,pred)>=base_score:
                logging.info(f"{gd} accuracy score for heat is {r2_score(y_test1,pred)}")
                score2.append(r2_score(y_test2,pred))
                model2.append(gd)
            lr=LinearRegression()
            logging.info(f"using {lr}")
            lr_params={"fit_intercept":[True,False]}
            gr=GridSearchCV(estimator=lr,param_grid=lr_params)
            gr.fit(x_train2,y_train2)
            fit_intercept_gr=gr.best_params_
            logging.info(f"best params {fit_intercept_gr} for {lr}")
            lr=LinearRegression(fit_intercept=fit_intercept_gr)
            lr.fit(x_train2,y_train2)
            pred=lr.predict(x_test2)
            if r2_score(y_test2,pred)>=base_score:
                logging.info(f"{lr} accuracy score for heat is {r2_score(y_test2,pred)}")
                score2.append(r2_score(y_test2,pred))
                model2.append(lr)

            dr=DecisionTreeRegressor()
            logging.info(f"using {dr}")
            dr_params={"criterion":['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],"splitter":["best", "random"],
            "min_samples_split":[1,2,3,4,5,6,7,8],"max_features":['auto', 'sqrt', 'log2']}
            gr=GridSearchCV(estimator=dr,param_grid=dr_params)
            gr.fit(x_train2,y_train2)
            dr_gr=gr.best_params_
            logging.info(f"best params {dr_gr} for {gr}")
            dr=DecisionTreeRegressor(criterion=dr_gr["criterion"],splitter=dr_gr["splitter"],min_samples_split=dr_gr["min_samples_split"],max_features=dr_gr["max_features"])
            dr.fit(x_train2,y_train2)
            pred=dr.predict(x_test2)
            if r2_score(y_test2,pred)>=base_score:
                logging.info(f"{lr} accuracy score for cool is {r2_score(y_test2,pred)}")
                score2.append(r2_score(y_test2,pred))
                model2.append(dr)
            rr=RandomForestRegressor()
            logging.info(f"using {rr}")
            rr_params={"criterion":['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],"min_samples_split":[1,2,3,4,5,6,7,8],"max_features":['sqrt', 'log2']}
            gr=GridSearchCV(estimator=rr,param_grid=rr_params)
            gr.fit(x_train2,y_train2)
            rr_gr=gr.best_params_
            logging.info(f"best params {rr_gr} for {rr}")
            rr=RandomForestRegressor(criterion=rr_gr["criterion"],min_samples_split=rr_gr["min_samples_split"],max_features=rr_gr["max_features"])
            rr.fit(x_train2,y_train2)
            pred=rr.predict(x_test2)
            if r2_score(y_test2,pred)>=base_score:
                logging.info(f"{rr} accuracy score for cool is {r2_score(y_test2,pred)}")
                score2.append(r2_score(y_test2,pred))
                model2.append(rr)
            return model1,score1,score2,model2
        except Exception as e:
            raise energy_exception(e,sys) from e

        









