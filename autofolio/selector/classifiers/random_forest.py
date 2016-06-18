import numpy as np
import pandas as pd

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace import Configuration

from autofolio.data.aslib_scenario import ASlibScenario

from sklearn.ensemble import RandomForestClassifier

__author__ = "Marius Lindauer"
__license__ = "BSD"


class RandomForest(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace):
        '''
            adds parameters to ConfigurationSpace 
        '''
        try:
            classifier = cs.get_hyperparameter("classifier")
            classifier.choices.append("RandomForest")
        except KeyError:
            classifier = CategoricalHyperparameter(
                "classifier", choices=["RandomForest"], default="RandomForest")
            cs.add_hyperparameter(classifier)

        n_estimators = UniformIntegerHyperparameter(
            name="rf:n_estimators", lower=10, upper=100, default=10, log=True)
        cs.add_hyperparameter(n_estimators)
        criterion = CategoricalHyperparameter(
            name="rf:criterion", choices=["gini", "entropy"], default="gini")
        cs.add_hyperparameter(criterion)
        max_depth = UniformIntegerHyperparameter(
            name="rf:max_depth", lower=10, upper=2**31, default=2**31, log=True)
        cs.add_hyperparameter(max_depth)
        min_samples_split = UniformIntegerHyperparameter(
            name="rf:min_samples_split", lower=2, upper=100, default=2, log=True)
        cs.add_hyperparameter(min_samples_split)
        bootstrap = CategoricalHyperparameter(
            name="rf:bootstrap", choices=[True, False], default=True)
        cs.add_hyperparameter(bootstrap)

        cond = InCondition(
            child=n_estimators, parent=classifier, values=["RandomForest"])
        cs.add_condition(cond)
        cond = InCondition(
            child=criterion, parent=classifier, values=["RandomForest"])
        cs.add_condition(cond)
        cond = InCondition(
            child=max_depth, parent=classifier, values=["RandomForest"])
        cs.add_condition(cond)
        cond = InCondition(
            child=min_samples_split, parent=classifier, values=["RandomForest"])
        cs.add_condition(cond)
        cond = InCondition(
            child=bootstrap, parent=classifier, values=["RandomForest"])
        cs.add_condition(cond)

    def __init__(self):
        '''
            Constructor
        '''

        self.model = None

    def __str__(self):
        return "RandomForest"

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

        self.model = RandomForestClassifier(n_estimators=config["rf:n_estimators"],
                                            criterion=config["rf:criterion"],
                                            max_depth=config["rf:max_depth"],
                                            min_samples_split=config[
                                                "rf:min_samples_split"],
                                            bootstrap=config["rf:bootstrap"],
                                            random_state=12345)
        self.model.fit(X, y, weights)

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

        return self.model.predict(X)
