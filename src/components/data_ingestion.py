import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


class DataIngestion:
    def __init__(self):
        self.artifact_dir = os.path.join("artifacts", "data_ingestion")
        os.makedirs(self.artifact_dir, exist_ok=True)

        self.raw_data_path = os.path.join(self.artifact_dir, "raw.csv")
        self.train_data_path = os.path.join(self.artifact_dir, "train.csv")
        self.test_data_path = os.path.join(self.artifact_dir, "test.csv")

    def initiate_data_ingestion(self):
        try:
            logging.info("========== Data Ingestion Started ==========")

            source_data_path = os.path.join("notebook", "data", "stud.csv")
            logging.info(f"Reading dataset from: {source_data_path}")

            df = pd.read_csv(source_data_path)

            logging.info("Saving raw data")
            df.to_csv(self.raw_data_path, index=False)

            logging.info("Performing train-test split")
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            logging.info("Saving train and test datasets")
            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logging.info("========== Data Ingestion Completed ==========")

            return (
                self.train_data_path,
                self.test_data_path
            )

        except Exception as e:
            logging.error("Exception occurred in data ingestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    DataIngestion().initiate_data_ingestion()
