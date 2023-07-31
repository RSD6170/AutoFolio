import logging
from concurrent import futures

import numpy as np
import psutil
from ConfigSpace import Configuration
from ConfigSpace import ConfigurationSpace
from ConfigSpace import InCondition

from autofolio.aslib.aslib_scenario import ASlibScenario

__author__ = "Marius Lindauer"
__license__ = "BSD"


class IndRegression(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace):
        '''
            adds parameters to ConfigurationSpace 
        '''

        selector = cs.get_hyperparameter("selector")
        if "IndRegressor" in selector.choices:
            return "regressor", "IndRegressor"
        else:
            return None, None

    def __init__(self, regressor_class):
        '''
            Constructor
        '''
        self.regressors = []
        self.logger = logging.getLogger("IndRegressor")
        self.regressor_class = regressor_class

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
        self.logger.info("Fit PairwiseRegressor with %s" %
                         (self.regressor_class))

        self.algorithms = scenario.algorithms

        n_algos = len(scenario.algorithms)
        X = scenario.feature_data.values

        regressor_tmp = [None] * n_algos

        with futures.ProcessPoolExecutor(max_workers=len(psutil.Process().cpu_affinity()) - 1) as e:
            fs = {e.submit(self.fit_instance, self.regressor_class, config, X, scenario.performance_data[scenario.algorithms[i]].values): i for i in range(n_algos)}
            for f in futures.as_completed(fs):
                regressor_tmp[fs[f]] = f.result()
            self.regressors = regressor_tmp

    @staticmethod
    def fit_instance(regressor_class, config: Configuration, x, y):
        reg = regressor_class(1)
        reg.fit(x, y, config)
        return reg

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
        scores = np.zeros((X.shape[0], n_algos))

        with futures.ProcessPoolExecutor(max_workers=len(psutil.Process().cpu_affinity()) - 1) as e:
            fs = {e.submit(self.predict_instance, self.regressors[i], X): i for i in range(n_algos)}
            for f in futures.as_completed(fs):
                scores[:, fs[f]] += f.result()

        # self.logger.debug(
        #   sorted(list(zip(scenario.algorithms, scores)), key=lambda x: x[1], reverse=True))
        algo_indx = np.argmin(scores, axis=1)

        schedules = dict((str(inst), [s]) for s, inst in
                         zip([(scenario.algorithms[i], cutoff + 1) for i in algo_indx], scenario.feature_data.index))
        # self.logger.debug(schedules)
        return schedules

    @staticmethod
    def predict_instance(reg, x):
        return reg.predict(x)

    def get_attributes(self):
        '''
            returns a list of tuples of (attribute,value) 
            for all learned attributes
            
            Returns
            -------
            list of tuples of (attribute,value) 
        '''
        reg_attr = self.regressors[0].get_attributes()
        attr = [{self.regressor_class.__name__: reg_attr}]

        return attr
