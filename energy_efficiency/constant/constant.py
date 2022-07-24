import os
import datetime

ROOT_DIR="E:\\Ineuron\\Project\\energy_efficiency\\energy-ML_project"
DUMP_DATA_LINE="E:\\Ineuron\\Project\\energy_efficiency\\energy-ML_project\\energy_efficiency\\artifact\\data_ingestion"
SCHEMA_FILE_PATH="E:\\Ineuron\\Project\\energy_efficiency\\energy-ML_project\\config\\schema.yaml"
VALIDATE_PATH="E:\\Ineuron\\Project\\energy_efficiency\\energy-ML_project\\energy_efficiency\\artifact\\data_validation"
CURRENT_TIMESTAMP=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
TRANSFORMED_PATH="E:\\Ineuron\\Project\\energy_efficiency\\energy-ML_project\\energy_efficiency\\artifact\\data_transformation\\transformed_data"
TRANSFORMED_PATH_PICKLE="E:\\Ineuron\\Project\\energy_efficiency\\energy-ML_project\\energy_efficiency\\artifact\\data_transformation\\preprocessed"
COOLING_TRAIN="E:\\Ineuron\\Project\\energy_efficiency\\energy-ML_project\\energy_efficiency\\artifact\\model_trainer\\cooling\\train"
COOLING_TEST="E:\\Ineuron\\Project\\energy_efficiency\\energy-ML_project\\energy_efficiency\\artifact\\model_trainer\\cooling\\test"
HEATING_TRAIN="E:\\Ineuron\\Project\\energy_efficiency\\energy-ML_project\\energy_efficiency\\artifact\\model_trainer\\heating\\train"
HEATING_TEST="E:\\Ineuron\\Project\\energy_efficiency\\energy-ML_project\\energy_efficiency\\artifact\\model_trainer\\heating\\test"

