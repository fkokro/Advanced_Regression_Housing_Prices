import sys
from dataclasses import dataclass
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from src.exception import CustomException
from src.logger import logging
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import save_object


@dataclass
class DataTransfomationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preporcessor.pkl')
    
class DataTransfomation:
    def __init__(self):
        self.data_transformation_config=DataTransfomationConfig()

    def get_data_encoder_object(self):
        """This function is responsible for data transformation"""
        try:
            numerical_columns = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'FirstFlrSF', 'SecondFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ThreeSsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

            categorical_columns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

            
            num_pipeline = Pipeline(
                steps=[
                    ('pass', 'passthrough')
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('one_hot_encoder', OneHotEncoder()),
                ]
            )
            
            logging.info('Caterorical columns encoding completed')
            
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
    
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, data_path):
        
        try:
            data_df=pd.read_csv(data_path, sep='|')
            target = data_df[['SalePrice']].copy()
            features = data_df.drop(columns=['SalePrice'])
            logging.info('Data load complete.')
            logging.info('Obtaining preprocessing objects')

            encoding_obj=self.get_data_encoder_object()          
            encoded_arr = encoding_obj.fit_transform(features)
            logging.info('Encoding complete.')
          
            df = pd.DataFrame(encoded_arr.toarray())

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_target, test_target = train_test_split(target, test_size=0.2, random_state=42)
            logging.info('Train and test data split complete.')

            scaler = StandardScaler(with_mean=False)
            scaled_training_data = scaler.fit_transform(train_set)
            scaled_testing_data = scaler.transform(test_set)
            logging.info('Data scaling complete.')

            processed_train_data_arr = np.column_stack(
                (scaled_training_data, np.array(train_target).reshape(-1,1))
                )


            processed_test_data_arr = np.column_stack(
                (scaled_testing_data, np.array(test_target).reshape(-1,1))
                )
            logging.info('Processed train and test objects complete.')
            
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=encoding_obj,
            )
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=scaler               
            )
            
            return(
                processed_train_data_arr,
                processed_test_data_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
              
        except Exception as e:
            raise CustomException(e,sys)
            