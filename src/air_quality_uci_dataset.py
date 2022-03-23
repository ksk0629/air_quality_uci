import datetime
from typing import List, Optional, Union

import numpy as np
import pandas as pd


class AirQualityUciDataset():
    """Air quality uci dataset class"""

    def __init__(self, dataset_path: str="AirQualityUCI.csv", this_type: Optional[Union[List[str], str]]=None) -> None:
        """
        :param str dataset_path: path to air quality uci dataset, defaults to "AirQualityUCI.csv"
        :param Optional[Union[List[str], str]] this_type: string or list of strings to define elements of this dataset, defaults to None
        """
        self.xs = None
        self.ys = None
        self.data = pd.read_csv(dataset_path)
        self.__is_valid_name(this_type)
        self.__this_type = this_type
        self.__drop_data()
        self.numerical_data = self.data.drop(["Date", "Time"], axis=1)

    def __is_valid_name(self, name: Optional[Union[List[str], str]]) -> bool:
        """Check whether a given this_type is valid or not.

        :param Optional[Union[List[str], str]] name: string or list of string to be checked
        """
        if name is None:
            return None
        elif type(name) is list:
            for name_str in name:
                self.__is_valid_nane_str(name_str)
            return None
        else:
            self.__is_valid_nane_str(name)

    def __is_valid_nane_str(self, name_str: str) -> None:
        """Check whether a given name_str is valid or not.

        :param str name_str: string to be checked
        :raises KeyError: if a given name is not in data.columns
        """
        for column_name in self.data.columns:
            if name_str != column_name:
                msg = f"{name} is not in {[column for column in self.data.columns]}"
                raise KeyError(msg)

    def __drop_data(self) -> None:
        """Drop data columns except for "Date", "Time", and "self.__this_type".
        """
        if self.__this_type is not None:
            kept_elements = ["Date", "Time"]

            if type(self.__this_type) is str:
                kept_elements = ["Date", "Time", self.__this_type]
            elif type(self.__this_type) is list:
                for key in self.__this_type:
                    kept_elements.append(key)

            self.data = self.data.drop(list(set(self.data.columns) - set(kept_elements)), axis=1)

    def __getitem__(self, key: int) -> pd.core.series.Series:
        """
        :param int key: a key that implies the position of a row
        :return pd.core.series.Series: the data of the row
        """
        return self.data.iloc[key, :]

    @property
    def datetimes(self) -> np.ndarray:
        """
        :return np.ndarray: date data over all of data
        """
        datetimes = []
        for key in range(len(self.data)):
            datetimes.append(self.__get_datetime(key))
        return np.array(datetimes)

    def __get_datetime(self, key: int) -> datetime.datetime:
        """Get datetime located a given key.

        :param int key: position of data to be got
        :return datetime.datetime: datetime data combined Date and Time located a given key
        """
        date_str = str(self.data["Date"][key])
        time_str = str(self.data["Time"][key])
        combined_date_str = f"{date_str[:11]} {time_str}"
        combined_date_datetime = datetime.datetime.strptime(combined_date_str, "%Y-%m-%d %H:%M:%S")
        
        return combined_date_datetime

    def get_values(self, data_names: List[str]) -> np.ndarray:
        """Get one data sequences as numpy.ndarray.

        :param List[str] data_names: list of names of data in self.data
        :return np.ndarray: one data
        """
        self.__is_valid_name(data_names)
        return self.data[data_names].values
