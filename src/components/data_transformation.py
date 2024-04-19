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
            logging.info('Obtaining preprocessing objects')
            encoding_obj=self.get_data_encoder_object()          
            encoded_arr = encoding_obj.fit_transform(features)
            df = pd.DataFrame(encoded_arr.toarray())
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info('Read train and test data complete.')
            scaler = StandardScaler(with_mean=False)
            scaled_training_data = scaler.fit_transform(train_set)
            scaled_testing_data = scaler.transform(test_set)
            print(scaled_testing_data )
            # train_set_array = scaler_obj.fit_transform(train_set)
            # test_set_array = scaler_obj.transform(test_set)
            # print(train_set_array)
            # logging.info('Dataframe split test, train sets.')
            # print(train_set.columns)
            # print(test_set.columns)
            # logging.info('Obtaining preprocessing object')
            # preprocessor_obj=self.get_data_transformer_object()
            
            # target_column_name = 'SalePrice'
            # # Training set
            # target_feature_train_df=train_set[target_column_name]
            # input_feature_train_df=train_set.drop(columns=[target_column_name],axis=1)
            # # Test set
            # target_feature_test_df=test_set[target_column_name]           
            # input_feature_test_df=test_set.drop(columns=[target_column_name],axis=1)
            
            
            # logging.info("Applying preprocessing object on training dataframe and testing dataframe.")
            # input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            # input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            # logging.info("Preprocessing complete.")

            # processed_train_data_arr = np.column_stack(
            #     (input_feature_train_arr.toarray(), np.array(target_feature_train_df))
            #     )

            # print(processed_train_data_arr)

            # processed_test_data_arr = np.column_stack(
            #     (input_feature_test_arr.toarray(), np.array(target_feature_test_df))
            #     )

            # print(processed_test_data_arr)                                            
        except Exception as e:
            raise CustomException(e,sys)
            