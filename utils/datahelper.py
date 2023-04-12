import os
import re, warnings
from dataclasses import dataclass
import pandas as pd
from pandas import DataFrame, Series
from typing import Tuple
from numpy import ndarray
from collections.abc import Iterable
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from utils.colorhelper import ColorPalette

"""
COLOR SETTING

%matplotlib inline

THEME_STYLE = "darkgrid"
THEME_PALETT = "pastel"

sns.set_theme(palette="pastel")
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")

https://seaborn.pydata.org/generated/seaborn.set_theme.html
https://matplotlib.org/stable/tutorials/colors/colormaps.html
https://stackoverflow.com/questions/48958208/how-do-you-change-the-default-font-color-for-all-text-in-matplotlib/48958263#48958263

THEME_PALETT = "Dark2"
THEME_STYLE = "ticks"# "darkgrid"
BG_COLOR= "black" # "darkslategray" # "midnightblue"# "dimgray" #"black"
# BG_COLOR="grey"
TEXT_COLOR= "snow" #"lightgrey"
sns.set_theme(style=THEME_STYLE, palette=THEME_PALETT)
sns.set(rc={'axes.facecolor': BG_COLOR, 'figure.facecolor':BG_COLOR,
            'text.color': TEXT_COLOR, 'axes.labelcolor': TEXT_COLOR, 'xtick.color': TEXT_COLOR, 'ytick.color': TEXT_COLOR })
"""

"""
https://stackoverflow.com/questions/25238442/setting-plot-background-colour-in-seaborn

import seaborn
theme_style = "white"
seaborn.set_theme(style=theme_style)
# bg_color="back"
# seaborn.set(rc={'axes.facecolor': bg_color, 'figure.facecolor':bg_color})


matplotlab dark color https://gist.github.com/mwaskom/7be0963cc57f6c89f7b2

temporay styling: https://matplotlib.org/stable/tutorials/introductory/customizing.html#temporary-styling
"""


def current_dir():
    """get the absolute path of the current file directory"""
    current_path = os.path.dirname(os.path.dirname(__file__))
    return current_path

def current_dir_subpath(subpath: str):
    """
    @param subpath: "data/train.csv", no leading "/"
    """
    # replace the leading / with ""
    return os.path.join(current_dir(), re.sub(r"^\/+" , "" , subpath))

def assign_const_col(df: DataFrame, col_name: str, value: any) -> DataFrame:
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.assign.html
    # https://stackoverflow.com/questions/29517072/add-column-to-dataframe-with-constant-value
    return df.assign(col_name=value)

def na_columns_mask(df: DataFrame) -> Series:
    """
    this method returns column mask with column contains NA values marked as True
    Reference:
    https://stackoverflow.com/questions/36226083/how-to-find-which-columns-contain-any-nan-value-in-pandas-dataframe/36226137#36226137

    get the colume with values

    df.loc[:, na_columns_mask(df)]
    
    get the column names with mask
    df.columns[na_columns_mask(df)].to_list()
    """
    # DataFrame.isna() returns a Boolean DataFrame with very cell value indicates if it is a NA value
    # The any(axis=0) operator reduce the column series to a true or false value, if there is any true NA value   
    return df.isna().any(axis=0)

def na_columns(df: DataFrame, filter_cols=[]) -> list:
    """
    this method returns columns containing NaN values from the given DataFrame, 
    the returned columns with NaN values can be filtered by the filter_cols variable.
    @param df: DataFrame
    @param filter_cols: filter list for the returned column list contains NaN values
    return column list
    """
    # get the Index of columns containing NaN values.
    index_cols = df.columns[na_columns_mask(df)]
    if filter_cols is None or len(filter_cols) == 0:
        return index_cols.to_list()
    else:
        # filter the NaN column index.
        return index_cols[index_cols.isin(filter_cols)].to_list()

def mean_values(df: DataFrame, cols: Iterable, value_func: callable=lambda x: x) -> dict:
    """
    this method calculated the mean value of all numerical columns from the given DataFrame and 
    return these mean values with the appropriate numerical column name as key in a dict obj

    Examples: 
    cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    dict = mean_values(df, cols)

    > dict = {
        'Pclass': 2.294881588999236,
        'Age': 29.881137667304014,
        'SibSp': 0.4988540870893812,
        'Parch': 0.3850267379679144,
        'Fare': 33.29547928134557
      }

    @param df: DataFrame
    @param cols: cols which shall be included
    @param value_func: a function which can be used to tranform the mean values, e.g. round 
    """
    # cols.remove('PassengerId')
    stats_df = df.describe()
    feature_mean_series = stats_df.loc['mean', cols]

    return dict(map(
        value_func,
        zip(feature_mean_series.index, 
            feature_mean_series.values) 
        ))

def fill_values(df: DataFrame, values: dict) -> DataFrame:
    # fill with multiple column values https://pandas.pydata.org/docs/reference/api/pandas.Series.fillna.html
    new_df = df.fillna(value=values)
    return new_df if new_df is not None else df

# def fill_values(df: DataFrame, cols: Iterable, fill_values: Iterable) -> DataFrame:   
#     assert len(cols) == len(fill_values), f"cols and fill_values must has the same length"
#     values = dict(zip(cols, fill_values))
#     return fill_values(df=df, values=values)


def count_na_row(df: DataFrame, cols: Iterable = []) -> int:
    if cols is None or len(cols) == 0:
        cols_selected = df.columns
    else:
        cols_selected = cols
    # any(axis) count if there is any true isna values in a row         
    return df[cols_selected].isna().any(axis=1).value_counts().get(key=True)

def profile(df: DataFrame, title: str = "") -> None:
    """Profile the current dataframe"""
    shape = df.shape
    stats_df = df.describe()
    numerical_features = stats_df.columns.to_list()
    na_num_columns = na_columns(df=df, filter_cols=numerical_features)
    count_na_raw_num_features = count_na_row(df, numerical_features)
    print(
        f"{title}\n" + 
        f"Shape: {shape}\n" + 
        f"Numerical Columns: {numerical_features}\n"+
        f"Num. Columns {na_num_columns} has total no. of NaN : {count_na_raw_num_features}\n"
        f"{stats_df}"        
    )


def feature_correlation(df: DataFrame, label: str, threshold: float) -> Tuple[DataFrame, DataFrame]:
    """this method returns correlation dataframe , and high correlation features to the label"""
    corr_df = df.corr(numeric_only=True)
    high_corr_feature_df = corr_df[(corr_df[label] > threshold) | (corr_df[label] < -threshold)].loc[:, [label]]
    return corr_df, high_corr_feature_df


# def truncate(f:float, n: int) -> str:
#     """
#     Truncates/pads a float f to n decimal places without rounding
#     @param f: float
#     @param n: number of decimals to keep
#     """
#     # https://stackoverflow.com/questions/783897/how-to-truncate-float-values/783927#783927
#     s = '%.12f' % f
#     i, p, d = s.partition('.')
#     return '.'.join([i, (d+'0'*n)[:n]])


def fill_missing_values_with_mean(df: DataFrame, pop_df: DataFrame, filter_cols: list, filter_func: callable) -> Tuple[DataFrame, dict]:
    """
    @param df: DataFrame containing missing values to be filled
    @param filter_cols: filter the columns need to fill the NaN missing value
    @param fiter_func: used to manipulate the inputation values which is function of tuple lambda x,y -> x,y
    @param pop_df: the population dataframe used to calculate the mean values.
    """
    na_cols = na_columns(df, filter_cols)
    mean_values_dict = mean_values(pop_df, na_cols, value_func=filter_func)
    return fill_values(df, values=mean_values_dict), mean_values_dict

# def one_hot(df: DataFrame, cols:list) -> DataFrame:
#     return pd.get_dummies(df, prefix=[cols], columns=cols)


class DataVisualizer(ColorPalette):
    def __init__(self, dark_mode: bool =True):
        super().__init__(dark_mode=dark_mode)

    
    def display_feature_correlation(self, corr_df: DataFrame, apply_abs: bool=True) -> None:
        if apply_abs:
            corr = corr_df.apply(abs)
        else:
            corr = corr_df
        # rc https://matplotlib.org/stable/tutorials/introductory/customizing.html
        rc={'figure.figsize': [8, 7] } 
        # plt.figure(figsize=(8,7))
        # map = sns.heatmap(corr_df, annot=True, cmap="RdYlGn")   
     
        self._set_plot_style(
            lambda: sns.heatmap(corr, annot=True, cmap=self._cmp()),         
            rc=rc)


@dataclass
class KaggleData:
    train_path: str = ""
    test_path: str = ""
    label_col: str = ""

    def __post_init__(self):
        self.train_X = pd.DataFrame()
        self.train_y = pd.Series(dtype= float)
        self.test_X = pd.DataFrame()
    
    def _cache_empty(self):
        """ Test if all cached DataFrame and Series are empty"""
        # testing all the cached dataframe and series shall not be empty
        # https://stackoverflow.com/questions/42360956/what-is-the-most-pythonic-way-to-check-if-multiple-variables-are-not-none
        return None not in map(lambda x: x.empty, (self.train_X, self.train_y, self.test_X) )
    
    def _one_hot(self, cols:list):
        self.train_X = pd.get_dummies(self.train_X, prefix=cols, columns=cols)
        self.test_X = pd.get_dummies(self.test_X, prefix=cols, columns=cols)

    def _train_test_X(self):
        return pd.concat([self.train_X, self.test_X], ignore_index=True)

    def _load(self) -> Tuple[DataFrame, DataFrame]:
        try:
            train_X_y_df = pd.read_csv(self.train_path, sep=",", header=0)
            test_X_df = pd.read_csv(self.test_path, sep=",", header=0)
        except: 
            train_X_y_df = pd.DataFrame()
            test_X_df = pd.DataFrame() 
        return train_X_y_df, test_X_df
    
    def load(self, one_hot_cols:Iterable = None) -> Tuple[DataFrame, DataFrame, Series]:
        train_X_y_df, test_X_df = self._load()
        # train_X_y_df and test_X_df may have different size of column,
        # thus the mask must be calculated separately
        train_column_mask: ndarray = ~train_X_y_df.columns.isin([self.label_col])
        test_column_mask: ndarray = ~test_X_df.columns.isin([self.label_col])
        # cache to this object
        self.train_X = train_X_y_df.loc[:, train_column_mask]
        self.train_y = train_X_y_df[self.label_col]
        self.test_X = test_X_df.loc[:, test_column_mask]
        
        if one_hot_cols is not None and len(one_hot_cols) != 0:
            self._one_hot(one_hot_cols)
            
        return self.train_X, self.test_X, self.train_y
    
    def load_all(self, one_hot_cols: Iterable = None) -> DataFrame:
        # testing all the cached dataframe and series shall not be empty
        if self._cache_empty():
            _, _, _ = self.load(one_hot_cols)
        return self._train_test_X()
    
    @staticmethod
    def split(X: DataFrame, y: Series, test_size=0.2, random_state=10) -> Tuple[DataFrame, DataFrame, Series, Series]:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_valid, y_train, y_valid
    
    
    @staticmethod
    def exclude_cols(df:DataFrame, cols: list = []) -> DataFrame:
        cols_mask: ndarray = ~df.columns.isin(cols)
        return df.loc[:, cols_mask]
    

    @staticmethod
    def select_cols(df:DataFrame, cols: list = []) -> DataFrame:
        if cols is None or len(cols) == 0:
            return df
        else: 
            return df[cols]

    
    def _all_data_sets(self) -> DataFrame:
        if self._cache_empty():
            _, _, _ = self.load()
        all_dist_df = pd.DataFrame()    
        for df, location in zip(
            (self.train_X, self.test_X, self._train_test_X()),
            ("train", "test", "total")
        ):  
            # add the data_partition attribute to all data sample 
            all_dist_df = pd.concat([all_dist_df, df.assign(data_partition=location)])
        return all_dist_df


    def boxplot_dist(self, features=[], orient="v", marker="x", legend="lower right", dark_mode=True):
        """
        @param legend: 
            upper right
            upper left
            lower left
            lower right
            right
            center left
            center right
            lower center
            upper center
            center
        @param edgecolor: matplotlib color, css colors https://matplotlib.org/stable/gallery/color/named_colors.html    
        """
        if dark_mode:
            edge_color = "lightgray" # lightgray, snow # for box
            # https://matplotlib.org/stable/tutorials/colors/colormaps.html
            # seaborn matplotlib color palett: https://seaborn.pydata.org/tutorial/color_palettes.html
            palett = "bright" # deep, muted, pastel, bright, dark, and colorblind
            text_color= "snow" #"lightgrey"
            bg_color= "black"
            # theme_palett = "Dark2"
            # theme_style = "darkgrid" # "ticks"
            # bg_color= "black" # "darkslategray" # "midnightblue"# "dimgray" #"black", "grey"
            # text_color= "snow" #"lightgrey"
            # sns.set_theme(style=theme_style, palette=theme_palett)
            # sns.set(rc={'axes.facecolor': bg_color, 'figure.facecolor':bg_color,
            #             'text.color': text_color, 'axes.labelcolor': text_color, 
            #             'xtick.color': text_color, 'ytick.color': text_color })
        else:
            edge_color = "black"
            text_color= "black"
            bg_color= "white"
            palett =  "pastel" # bright        

        all_df = self._all_data_sets()
        all_num_cols = all_df.describe().columns.to_list() + ["data_partition"]
        if features is None or len(features)==0:
            cols = all_num_cols
        else:     
            cols_set = set(all_df.describe().columns.to_list())
            features_set = set(features)
            common_set = cols_set.intersection(features_set)
            common_set.add("data_partition")
            if len(common_set) == 1: 
                # only has the data_partition
                cols = all_num_cols
                warnings.warn(f"numerical features {features} are not found, displays all numerical features", category=UserWarning, stacklevel=1)
            else:    
                cols = list(common_set)
        
        # https://stackoverflow.com/questions/42004381/box-plot-of-a-many-pandas-dataframes/42005692#42005692
        # the pd.melt create the value and features column, where feature column encode all the feature   
        mdf = pd.melt(all_df[cols], id_vars=['data_partition'], var_name=["numerical_features"])
        """boxplot all numerical features for the different datasets 'train', 'test', 'total' """

        # make the edge of boxes white
        # https://stackoverflow.com/questions/43434020/black-and-white-boxplots-in-seaborn/65529178#65529178
        # 
        PROPS = {
            #'boxprops':{'facecolor':'none', 'edgecolor':'red'},
            'boxprops':{'edgecolor':edge_color},
            'medianprops':{'color':edge_color},
            'whiskerprops':{'color':edge_color},
            'capprops':{'color':edge_color},
            'flierprops' :{"marker" : marker, "markerfacecolor":edge_color, "markeredgecolor": edge_color},
            'palette': palett
        }

        def local_plot(orient: str, df:DataFrame, PROPS):
            match orient:
                case "h": 
                    # ax = sns.boxplot(y="data_partition", x="value", hue="numerical_features", data=mdf, flierprops={"marker": marker}, **PROPS)
                    ax = sns.boxplot(y="data_partition", x="value", hue="numerical_features", data=df, **PROPS)
                case "v", _:
                    # ax = sns.boxplot(x="data_partition", y="value", hue="numerical_features", data=mdf, flierprops={"marker": marker, }, **PROPS)
                    ax = sns.boxplot(x="data_partition", y="value", hue="numerical_features", data=df, **PROPS)


        
        if dark_mode:
            with plt.style.context('dark_background'):
                sns.set(rc={'axes.facecolor': bg_color, 'figure.facecolor':bg_color,
                    'text.color': text_color, 'axes.labelcolor': text_color, 
                    'xtick.color': text_color, 'ytick.color': text_color })
                local_plot(orient=orient, df=mdf, PROPS=PROPS)
        else:     
            local_plot(orient=orient, df=mdf, PROPS=PROPS)
        plt.legend(loc=legend)
        plt.show()        
            
           


