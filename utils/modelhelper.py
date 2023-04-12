"""
ClassifierMixin is a general type of Classification mode
"""

from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
from pandas import DataFrame, Series
from typing import Any
import random
import shap
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    f1_score, 
    roc_auc_score, # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    auc,
    roc_curve, 
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from matplotlib import pyplot as plt
from utils.colorhelper import ColorPalette

import warnings
from numpy import ndarray
from typing import Tuple, Callable

class ModelExplainer(ColorPalette):
    """
    customize SHAP plot color
    https://towardsdatascience.com/how-to-easily-customize-shap-plots-in-python-fdff9c0483f2?gi=754319eae2c8

    Save Explainer
    https://github.com/slundberg/shap/issues/295

    """
    def __init__(self, model: Any, data: DataFrame, dark_mode: bool = False, algorithm: str="auto"):
        super().__init__(dark_mode=dark_mode)
        """
        Note: For unsupported model use Explainer with model.predict and X feature data

        from sklearn_rvm import EMRVR
        model_rvr=EMRVR(kernel="linear").fit(X, y)
        explainer = shap.Explainer(model_rvr.predict, X)
        Reference: 
        https://github.com/slundberg/shap/issues/2399#issuecomment-1076701148
        
        Note: Difference between Explainer and KernelExplainer
        Kernel Explainer:
        https://shap-lrjball.readthedocs.io/en/latest/generated/shap.KernelExplainer.html
        Explainer: primary interface to call shap explainer, like a factory function
        https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html
        """
        # feature data without label
        self.data = data
        # self.explainer = shap.Explainer(model, algorithm=algorithm)

        # explainer will choose an optimal explaining methods
        self.explainer = shap.Explainer(model=model.predict, masker=self.data, algorithm=algorithm)

        # KernelExplainer for any model, model agnostic 
        # https://github.com/slundberg/shap#model-agnostic-example-with-kernelexplainer-explains-any-function
        

        # this call may took some time
        self.shap_values = self.explainer(self.data)              
    

    def summary_plot(self) -> None:
        """
        plot_type: dot, bar or violin
        https://shap-lrjball.readthedocs.io/en/latest/generated/shap.summary_plot.html
        """
        def inner():
            if self.dark_mode:
                text_color= "snow" #"lightgrey"
            else:
                text_color= "black"
            rc={'text.color': text_color, 'axes.labelcolor': text_color, 
                    'xtick.color': text_color, 'ytick.color': text_color }     
            with plt.rc_context(rc=rc):
                # not using the plot_type since default for single and multi output are choosen automatically
                shap.summary_plot(self.shap_values, self.data, cmap=self._cmp())
        return self._set_plot_style(inner)


    def beeswarm_plot(self) -> None:
        color="snow"
        # https://matplotlib.org/stable/tutorials/introductory/customizing.html 
        # rc = {'axes.facecolor': color, 'axes.labelcolor': color, 'xtick.color': color, 'ytick.color': color, 'text.color': color, 'xtick.labelcolor': color}
        rc = {'axes.edgecolor': color}
        self._set_plot_style(lambda:
            # shap.plots.beeswarm(self.shap_values, cmap=self._cmp())
            shap.plots.beeswarm(self.shap_values),
            rc=rc        
        )       

    
    def waterfall_plot(self, idx: int = 0) -> None:
        
        # @self.valid_index
        # def inner(df: DataFrame, idx: int):
        #     with plt.style.context('dark_background'):
        #         shap.plots.waterfall(df[idx])

        # return inner(self.shap_values, idx)
        self._set_plot_style(lambda: 
               shap.plots.waterfall(self.shap_values[idx])               
        )
             

    def scatter_plot(self, feature: str, interact_feature: str = None, auto_interact_features: int = None) -> None:
        """
        param: auto_interact_features the number of max interact_features need to be chosen automatically
        
        """
        if interact_feature is None and auto_interact_features is not None:
            # use shap.approximate_interactions to guess which features may interact with age.
            inds = shap.utils.potential_interactions(self.shap_values[:, feature], self.shap_values)
            for i in range(auto_interact_features):
                self._set_plot_style(lambda:
                    # scatter plot api https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/scatter.html                 
                    shap.plots.scatter(self.shap_values[:, feature] , color=self.shap_values[:, inds[i]])
                )
        else:
            self._set_plot_style(lambda: 
                    shap.plots.scatter(self.shap_values[:, feature] , color=self.shap_values[:, interact_feature])
                )          
    

    # @staticmethod
    def valid_index(func: Callable) -> Callable:
        """
        This static method is used to help testing the idx arg for an arbitrary Callable
        is in the range of 0 and max size of a DataFrame arg for the same arbitrary Callable

        function factory example: https://github.com/kubeflow/pipelines/blob/master/sdk/python/kfp/components/component_factory.py#L429
        partial function example of functool module: https://www.geeksforgeeks.org/partial-functions-python/
        variable passing in decorator: https://www.programiz.com/python-programming/decorator        
        """
        def inner(*args, **kwargs):
            # testing whether the decorated function was called with foo(df=pd.DataFrame(), idx=0)
            if "df" in kwargs and "idx" in kwargs:
                df = kwargs["df"]
                idx = kwargs["idx"]
                max = df.shape[0]
                if 0 <= idx < max:
                    # call return to pass back the result of Callable
                    return func(*args, **kwargs)
                else: 
                    warnings.warn(f"idx must between [0, {max})", category=UserWarning)
        return inner 
    
  
    def force_plot(self, idx: int, use_expected_value: bool = False) -> None:
        """
        show force plot of a data input 
        @param idx: row index position of the explainer data 
        """
        # max = self.data.shape[0] 
        # if 0 <= idx < max:
        #     # set the matplotlib flag so that plot is generated by matplotlib instead of js
        #     shap.plots.force(self.shap_values[idx], matplotlib=True)
        # else: 
        #     warnings.warn(f"idx must between [0, {max})", category=UserWarning)

        # @self.valid_index
        @ModelExplainer.valid_index
        def inner(df: DataFrame, idx: int):
            # set the matplotlib flag so that plot is generated by matplotlib instead of js
            if use_expected_value:
                # used for kernelExplainer
                # self is send by the @self.valid_index
                self._set_plot_style(lambda:
                    # need to call shap.inijs()
                    shap.plots.force(self.explainer.expected_value, self.shap_values[idx], self.data.iloc[idx], link="logit")              
                    # shap.plots.force(self.explainer.expected_value, self.shap_values[idx][0,:], self.data.iloc[0,:], link="logit")
                )
            else:    
                self._set_plot_style(lambda:                 
                    shap.plots.force(self.shap_values[idx], matplotlib=True)
                )      
            
        # the decorated valid_index function must be called with kwarg df and idx
        return inner(df=self.data, idx=idx)
    

class ModelKernelExplainer(ModelExplainer):
    def __init__(self, model: Any, train_data: DataFrame, inference_data: DataFrame,                
                 dark_mode: bool = False, 
                 link="identity",
                 labels=[0,1], # int class label
                 ):
        """
        link function is set to be "identity" for the random model particular case, while using "l"
        """
        # super() is calling the immediate base class, so we need to call the parent parent class with name
        ColorPalette.__init__(self, dark_mode=dark_mode)
        
        self.data = inference_data
        self.link = link
        self.labels = labels
        # self.explainer = shap.Explainer(model, algorithm=algorithm)

        # explainer is a callable object for tree models
        """ Notice: it tooks time to train the KernelExpainer, not the calculation of shap_values """
        # switched to kernel method instead of auto
        # https://shap-lrjball.readthedocs.io/en/latest/generated/shap.KernelExplainer.html
        # Outputs: Using 712 background data samples could cause slower run times. Consider using shap.sample(data, K) or shap.kmeans(data, K) to summarize the background as K samples.
        
        # Initialize the explainer, and calculated the expected values
        self.explainer = shap.KernelExplainer(
            model=model.predict_proba, data=train_data.to_numpy(), link=link,
            algorithm="kernel"
            )

        # KernelExplainer for any model, model agnostic 
        # https://github.com/slundberg/shap#model-agnostic-example-with-kernelexplainer-explains-any-function
        

        # this call may took some time
        # self.nsamples = min(nsamples, self.data.shape[0])
        """nsamples are the batch size
        https://shap-lrjball.readthedocs.io/en/latest/generated/shap.KernelExplainer.html#shap.KernelExplainer.shap_values
        """
        # self.shap_values = self.explainer.shap_values(self.data.to_numpy(), nsamples=50)
        # outputs: the progressbar to calculate the shap_values
        # we can also just pass X_valid.iloc[[0]] one row to calculate one shap value.
        '''Build a weighted linear regression model, for the local data to calculate shap value
        This takes very much time
        code of the kernel explainer
        https://github.com/slundberg/shap/blob/master/shap/explainers/_kernel.py
        '''
        self.shap_values = self.explainer.shap_values(self.data.to_numpy(), nsamples="auto")
        

    def force_plot(self, idx: int) -> None:
        # super().force_plot(idx=idx, use_expected_value=True)
        # print(self.label)
        # print(idx)
        # print(self.link)
        for label in self.labels:
            print(f"\npredicted probability for label {label} == {'Perisched' if label == 0 else 'Survived'}")
            shap.plots.force(self.explainer.expected_value[label], 
                             self.shap_values[label][idx], self.data.iloc[idx], link=self.link, matplotlib=True)
        
        
        #shap.plots.force(explainer.explainer.expected_value[1], 
        #         explainer.shap_values[1][0], explainer.data.iloc[0], link="identity")

        # output for all instance                       
        #  shap.plots.force(self.explainer.expected_value[self.label], 
        #                   self.shap_values[self.label][idx], self.data.iloc[idx], link=self.link) 


    def waterfall_plot(self, idx: int = 0) -> None:
        # since the model.predict_proba is used, so that logits are passed and shap_values returns a list for all the logits
        # self._set_plot_style(lambda: 
        #        shap.plots.waterfall(self.shap_values[self.label][idx])               
        # )
        
        # no waterfall for kernelExplainer, since the shap_values are list or array and not Explainer
        pass
    

    def beeswarm_plot(self) -> None:
        """override the ModelExplainer, no beeswarm for kernelExplainer"""
        pass

    def scatter_plot(self, feature: str, interact_feature: str = None, class_label: int = 1, **kwargs) -> None:
        """
        using **kwargs allow max_interact_feature: int input

        added additonal class_label defaut to the base class scatter_plot
        example:
        explainer.scatter_plot(feature="Age", interaction_feature="Sex_female")
        """
        if "auto_interact_features" in kwargs:
            warnings.warn("auto_interact_features is ignored.")

        if class_label is not None:
            labels = [class_label]
        else:
            labels = self.labels

        for label in labels:
            print(f"\npredicted probability for label {label} == {'Perisched' if label == 0 else 'Survived'}")
            if interact_feature is not None:
                self._set_plot_style(lambda:
                    shap.dependence_plot(feature, self.shap_values[label], self.data, interaction_index=interact_feature)
                )
            else:
                # get interaction index automatically
                self._set_plot_style(lambda:
                    shap.dependence_plot(feature, self.shap_values[label], self.data)                      
                )  
                      

class ModelValidator(ColorPalette):
    """Binary Classification Model Validator"""
    def __init__(self, y_truth: Series, y_pred: ndarray, label_name_map: dict={0:"Perished", 1: "Survived"}, pos_label=1, dark_mode: bool=False):
        super().__init__(dark_mode=dark_mode)

        self.y_truth = y_truth
        self.y_pred = y_pred
        self.label_name_map = label_name_map
        self.unknown_label_name = "unknown"
        # value list of categorical labels 
        self.cat_int_labels = self.y_truth.value_counts().index.to_list()
        # name list of categorical labels
        self.cat_char_labels = [ self.label_name_map.get(idx, self.unknown_label_name) for idx in self.cat_int_labels ]
        # positive label value for binary classification
        self.pos_label = pos_label


    def confusion_matrix(self) -> Tuple[DataFrame, ndarray]:
        """
        returns confusion matrix as a DataFrame
        with Columns and Row index of the Categorical labels
        """
        self.conf_mat = confusion_matrix(self.y_truth, self.y_pred, labels=self.cat_int_labels)
        # use both cat_char_labels as index and column names
        self.conf_mat_df = pd.DataFrame(self.conf_mat, index=self.cat_char_labels, columns=self.cat_char_labels)
        return self.conf_mat_df, self.conf_mat
    

    def accuracy_score(self):
        self.acc = accuracy_score(self.y_truth, self.y_pred)
        return self.acc
    

    def f1_score(self):
        self.f1 = f1_score(self.y_truth, self.y_pred)
        return self.f1
    
    
    def roc_curve(self)-> Tuple[ndarray, ndarray, ndarray, float]:
        """
        ROC: Receiver operating characteristic
        positive label is 1 is the survived label
        
        Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
        https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py
        
        AUC: The Area Under the Curve (AUC) is the measure of the ability of a binary classifier to distinguish between classes
        and is used as a summary of the ROC curve.
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
        
        """
        # fpr, tpr, thresholds = roc_curve(self.y_truth, self.y_pred, pos_label=)
        self.roc_fpr, self.roc_tpr, self.roc_thresholds = roc_curve(self.y_truth, self.y_pred, pos_label=self.pos_label)
        self.roc_auc = auc(self.roc_fpr, self.roc_tpr)
        return self.roc_fpr, self.roc_tpr, self.roc_thresholds, self.roc_auc
    
    def evaluate(self) -> dict:
        self.confusion_matrix()
        self.f1_score()
        self.accuracy_score()
        self.roc_curve()

        return {
            "conf_mat_df": self.conf_mat_df,
            "conf_mat": self.conf_mat,
            "f1": self.f1,
            "acc": self.acc,
            "roc_fpr": self.roc_fpr,
            "roc_tpr": self.roc_tpr,
            "roc_thresholds": self.roc_thresholds,
            "auc": self.roc_auc 
        }
    

    @staticmethod
    def print_eval_result(result: dict)-> None:
        """helper method to print the eval result"""
        for key, value in result.items():
            print(f"## {key}:")
            print(f"{value}\n")


    def display_roc_curve(self):
        """
        https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py
        """
        def inner():
            roc_display = RocCurveDisplay(
                fpr=self.roc_fpr, tpr=self.roc_tpr,roc_auc=self.roc_auc, 
                 pos_label=self.pos_label).plot()
            return roc_display    
        return self._set_plot_style(inner)
    

    def display_confusion_matrix(self):
        def inner():
            cm_display = ConfusionMatrixDisplay(self.conf_mat, display_labels=self.cat_char_labels).plot()
            return cm_display      
        return self._set_plot_style(inner)  



class ProbBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    this Classifier learns a Maximal Likelihood as Probability based on single feature associated with positive label,
    this probability is used to predict for a binary classifier 
    """
    @staticmethod
    def _decision(probabilty: float) -> int:
        """
        0, 1 decision making based on probability
        decorated with staticmethod so that it can also be called with self._decision 
        without self being passed as the first param implicity
        
        Reference:
        https://stackoverflow.com/questions/43587044/do-we-really-need-staticmethod-decorator-in-python-to-declare-static-method/43587154#43587154
        """
        return int(random.random() <= probabilty)
    

    @staticmethod
    def feature_position(df: DataFrame, col_name):
        """
        returns the index position of a given column name for a DataFrame.
        This is a helper method used for init a model 
        """
        return df.columns.get_loc(col_name)
    

    def __init__(self, feature_position, feature_value):
        # position in the training data for a categorical feature
        self.feature_position = feature_position
        # the true positive values of the categorical feature
        self.feature_value = feature_value
        self.num_classes = 2


    def fit(self, X, y=None):
        # print(type(X))
        # print(type(y))
        X_y_df = pd.concat([X, y], axis=1)
        # get the probability of positive
        feature_positive_label = X_y_df.loc[X_y_df.iloc[:, self.feature_position] == self.feature_value].iloc[:, -1]
        # get the random rate
        self._rate = sum(feature_positive_label) / len(feature_positive_label)
        # round two decimal digists
        self._rate = round(self._rate, 2)


    def predict(self, X):
        # both numpy.ndarray and pandas.DataFrame has .shape property
        # get 0 or 1 by probability, the dim 0 size is the same as input
        return np.array([self._decision(self._rate) for _ in range(0, X.shape[0])])
    

    def predict_proba(self, X, y=None):
        """get the logits using 
        num_class = 2
        np.eye(num_class)[a]

        import numpy as np
        predicted_labels = np.array([1, 0, 1, 0])
        num_class = 2
        np.eye(num_class)[predicted_labels]

        return:
        array([
         [0., 1.],
         [1., 0.],
         [0., 1.],
         [1., 0.]
         ])
        """
        predicted_labels = self.predict(X)
        # print(predicted_labels)
        return np.eye(self.num_classes)[predicted_labels]