import logging
from concurrent import futures

import numpy as np
import psutil
from ConfigSpace import Configuration
from ConfigSpace import ConfigurationSpace
from ConfigSpace import InCondition
from sklearn.preprocessing import MinMaxScaler

from autofolio.aslib.aslib_scenario import ASlibScenario

__author__ = "Marius Lindauer"
__license__ = "BSD"


class PairwiseClassifier(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace):
        '''
            adds parameters to ConfigurationSpace 
        '''

        selector = cs.get_hyperparameter("selector")
        if "PairwiseClassifier" in selector.choices:
            return "classifier", "PairwiseClassifier"
        else:
            return None, None

    def __init__(self, classifier_class):
        '''
            Constructor
        '''
        self.classifiers = {}
        self.logger = logging.getLogger("PairwiseClassifier")
        self.classifier_class = classifier_class
        self.normalizer = MinMaxScaler()

    def fit(self, scenario: ASlibScenario, config: Configuration):
        '''
            fit pca object to ASlib scenario data

            Arguments
            ---------
            scenario: data.aslib_scenario.ASlibScenario
                ASlib Scenario with all data in pandas
            config: ConfigSpace.Configuration
                configuration
        '''
        self.logger.info("Fit PairwiseClassifier with %s" %
                         (self.classifier_class))

        self.algorithms = scenario.algorithms

        n_algos = len(scenario.algorithms)
        X = scenario.feature_data.values
        # since sklearn (at least the RFs) 
        # uses float32 and we pass float64,
        # the normalization ensures that floats
        # are not converted to inf or -inf
        # X = (X - np.min(X)) / (np.max(X) - np.min(X))
        X = self.normalizer.fit_transform(X)


        with futures.ProcessPoolExecutor(max_workers=len(psutil.Process().cpu_affinity()) - 1) as e:
            fs = {e.submit(self.fit_instance, self.classifier_class, config, X, scenario.performance_data[scenario.algorithms[i]].values, scenario.performance_data[scenario.algorithms[j]].values): (i,j) for i in range(n_algos) for j in range(i+1, n_algos)}
            for f in futures.as_completed(fs):
                self.classifiers[fs[f]] = f.result()

    @staticmethod
    def fit_instance(classifier_class, config, X, y_i, y_j):
        y = y_i < y_j
        weights = np.abs(y_i - y_j)
        clf = classifier_class(1)
        clf.fit(X, y, config, weights)
        return clf

    def predict(self, scenario: ASlibScenario):
        '''
            predict schedules for all instances in ASLib scenario data

            Arguments
            ---------
            scenario: data.aslib_scenario.ASlibScenario
                ASlib Scenario with all data in pandas

            Returns
            -------
                schedule: {inst -> (solver, time)}
                    schedule of solvers with a running time budget
        '''

        if scenario.algorithm_cutoff_time:
            cutoff = scenario.algorithm_cutoff_time
        else:
            cutoff = 2 ** 31

        n_algos = len(scenario.algorithms)
        X = scenario.feature_data.values
        X = self.normalizer.transform(X)
        scores = np.zeros((X.shape[0], n_algos))

        with futures.ProcessPoolExecutor(max_workers=len(psutil.Process().cpu_affinity()) - 1) as e:
            fs = {e.submit(self.predict_instance, self.classifiers[(i,j)], X): (i,j) for i in range(n_algos) for j in range(i + 1, n_algos)}
            for f in futures.as_completed(fs):
                i,j = fs[f]
                Y = f.result()
                scores[Y == 1, i] += 1
                scores[Y == 0, j] += 1

        # self.logger.debug(
        #   sorted(list(zip(scenario.algorithms, scores)), key=lambda x: x[1], reverse=True))
        algo_indx = np.argmax(scores, axis=1)

        schedules = dict((str(inst), [s]) for s, inst in
                         zip([(scenario.algorithms[i], cutoff + 1) for i in algo_indx], scenario.feature_data.index))
        # self.logger.debug(schedules)
        return schedules
    @staticmethod
    def predict_instance(clf, X):
        return clf.predict(X)

    def get_attributes(self):
        '''
            returns a list of tuples of (attribute,value) 
            for all learned attributes
            
            Returns
            -------
            list of tuples of (attribute,value) 
        '''
        class_attr = self.classifiers[0].get_attributes()
        attr = [{self.classifier_class.__name__: class_attr}]

        return attr
