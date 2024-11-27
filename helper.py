import calendar
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def convert_to_month_name(year_month):

    if year_month == 'Summe':
        return 'Summe'
    month = str(year_month)[4:6]


    month_name = calendar.month_name[int(month)]
    return month_name


def transform_new_data(new_data, encoder, original_one_hot_columns):

    new_data_copy = new_data.copy()


    encoded_columns = encoder.transform(new_data_copy[original_one_hot_columns])


    encoded_column_names = encoder.get_feature_names_out(original_one_hot_columns)


    encoded_df = pd.DataFrame(
        encoded_columns,
        columns=encoded_column_names,
        index=new_data_copy.index
    )


    result_df = pd.concat([
        new_data_copy.drop(columns=original_one_hot_columns),
        encoded_df
    ], axis=1)

    return result_df