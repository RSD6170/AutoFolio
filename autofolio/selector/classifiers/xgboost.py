import numpy as np
import psutil
import xgboost as xgb
from ConfigSpace import Configuration
from ConfigSpace import ConfigurationSpace
from ConfigSpace import InCondition
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter

__author__ = "Marius Lindauer"
__license__ = "BSD"


NUM_ROUND = 50
ALPHA = 1
LAMBDA = 1
COLSAMPLE_BYLEVEL = 1
COLSAMPLE_BYTREE = 1
SUBSAMPLE = 1
MAX_DELTA_STEP = 0
MIN_CHILD_WEIGHT = 1
MAX_DEPTH = 6
GAMMA = 0
ETA = 0.3


class XGBoost:

    @staticmethod
    def add_params(cs: ConfigurationSpace):
        '''
            adds parameters to ConfigurationSpace 
        '''

        try:
            classifier = cs.get_hyperparameter("classifier")
            if "XGBoost" not in classifier.choices:
                return

            num_round = UniformIntegerHyperparameter(
                name="xgb:num_round", lower=10, upper=100, default_value=NUM_ROUND, log=True)
            cs.add_hyperparameter(num_round)
            alpha = UniformFloatHyperparameter(
                name="xgb:alpha", lower=0, upper=10, default_value=ALPHA)
            cs.add_hyperparameter(alpha)
            lambda_ = UniformFloatHyperparameter(
                name="xgb:lambda", lower=1, upper=10, default_value=LAMBDA)
            cs.add_hyperparameter(lambda_)
            colsample_bylevel = UniformFloatHyperparameter(
                name="xgb:colsample_bylevel", lower=0.5, upper=1, default_value=COLSAMPLE_BYLEVEL)
            cs.add_hyperparameter(colsample_bylevel)
            colsample_bytree = UniformFloatHyperparameter(
                name="xgb:colsample_bytree", lower=0.5, upper=1, default_value=COLSAMPLE_BYTREE)
            cs.add_hyperparameter(colsample_bytree)
            subsample = UniformFloatHyperparameter(
                name="xgb:subsample", lower=0.01, upper=1, default_value=SUBSAMPLE)
            cs.add_hyperparameter(subsample)
            max_delta_step = UniformFloatHyperparameter(
                name="xgb:max_delta_step", lower=0, upper=10, default_value=MAX_DELTA_STEP)
            cs.add_hyperparameter(max_delta_step)
            min_child_weight = UniformFloatHyperparameter(
                name="xgb:min_child_weight", lower=0, upper=20, default_value=MIN_CHILD_WEIGHT)
            cs.add_hyperparameter(min_child_weight)
            max_depth = UniformIntegerHyperparameter(
                name="xgb:max_depth", lower=1, upper=10, default_value=MAX_DEPTH)
            cs.add_hyperparameter(max_depth)
            gamma = UniformFloatHyperparameter(
                name="xgb:gamma", lower=0, upper=10, default_value=GAMMA)
            cs.add_hyperparameter(gamma)
            eta = UniformFloatHyperparameter(
                name="xgb:eta", lower=0, upper=1, default_value=ETA)
            cs.add_hyperparameter(eta)

            cond = InCondition(
                child=num_round, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=alpha, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=lambda_, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=colsample_bylevel, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=colsample_bytree, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=subsample, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=max_delta_step, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=min_child_weight, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=max_depth, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=gamma, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=eta, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
        except:
            return

    def __init__(self, jobs=len(psutil.Process().cpu_affinity())):
        '''
            Constructor
        '''

        self.model = None
        self.jobs = jobs
        self.attr = []
        xgb.set_config(verbosity=0)

    def __str__(self):
        return "XGBoost"

    def fit(self, X, y, config: Configuration, weights=None):
        '''
            fit pca object to ASlib scenario data

            Arguments
            ---------
            X: numpy.array
                feature matrix
            y: numpy.array
                label vector
            weights: numpy.array
                vector with sample weights
            config: ConfigSpace.Configuration
                configuration

        '''

        self.model = xgb.XGBRegressor(objective="binary:logistic",
                         n_estimators=config.get("xgb:num_round", NUM_ROUND),
                         reg_alpha=config.get("xgb:alpha", ALPHA),
                         reg_lambda=config.get("xgb:lambda", LAMBDA),
                         colsample_bylevel=config.get("xgb:colsample_bylevel", COLSAMPLE_BYLEVEL),
                         colsample_bytree=config.get("xgb:colsample_bytree", COLSAMPLE_BYTREE),
                         subsample=config.get("xgb:subsample", SUBSAMPLE),
                         max_delta_step=config.get("xgb:max_delta_step", MAX_DELTA_STEP),
                         min_child_weight=config.get("xgb:min_child_weight", MIN_CHILD_WEIGHT),
                         max_depth=config.get("xgb:max_depth", MAX_DEPTH),
                         gamma=config.get("xgb:gamma", GAMMA),
                         learning_rate=config.get("xgb:eta", ETA),
                         random_state=12345,
                         n_jobs=self.jobs)
        self.model.fit(X, y, sample_weight=weights)

    def predict(self, X):
        '''
            transform ASLib scenario data

            Arguments
            ---------
            X: numpy.array
                instance feature matrix

            Returns
            -------

        '''
        self.model.set_params(n_jobs=len(psutil.Process().cpu_affinity()) - 1)
        preds = np.array(self.model.predict(X))
        preds[preds < 0.5] = 0
        preds[preds >= 0.5] = 1
        return preds

    def get_attributes(self):
        '''
            returns a list of tuples of (attribute,value) 
            for all learned attributes

            Returns
            -------
            list of tuples of (attribute,value) 
        '''
        return self.attr
