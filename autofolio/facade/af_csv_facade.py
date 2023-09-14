import logging
import pickle

import numpy as np
from ConfigSpace.configuration_space import Configuration

from autofolio.aslib.aslib_scenario import ASlibScenario
from autofolio.autofolio import AutoFolio

__author__ = "Marius Lindauer"
__license__ = "BSD"
__version__ = "2.1.0"


class AFCsvFacade(object):

    def __init__(self,
                 scenario: ASlibScenario = None,
                 perf_fn: str = None,
                 feat_fn: str = None,
                 scenario_path: str = None,
                 objective: str = "solution_quality",
                 runtime_cutoff: float = None,
                 maximize: bool = True,
                 cv_fn: str = None,
                 seed: int = 12345
                 ):
        """ Constructor """
        if scenario is not None:
            self.scenario = scenario
        else:
            self.scenario = ASlibScenario()

            if not scenario_path is None:
                self.scenario.read_scenario(scenario_path)
            else:
                self.scenario.read_from_csv(perf_fn=perf_fn,
                                        feat_fn=feat_fn,
                                        objective=objective,
                                        runtime_cutoff=runtime_cutoff,
                                        maximize=maximize,
                                        cv_fn=cv_fn)
        self.seed = seed

        self.af = AutoFolio(random_seed=seed)
        self.logger = logging.getLogger("AF Facade")
        self.unpickeld = None

    def fit(self,
            config: Configuration = None,
            save_fn: str = None):
        """ Train AutoFolio on data from init"""
        self.logger.info("Fit")
        if config is None:
            cs = self.af.get_cs(self.scenario, {})
            config = cs.get_default_configuration()
        feature_pre_pipeline, pre_solver, selector = self.af.fit(scenario=self.scenario, config=config)

        if save_fn:
            self.af._save_model(save_fn, self.scenario, feature_pre_pipeline, pre_solver, selector, config)
        self.logger.info("AutoFolio model saved to %s" % (save_fn))
        self.unpickeld = self.scenario, feature_pre_pipeline, pre_solver, selector, config

    def tune(self,
             wallclock_limit: int = np.inf,
             runcount_limit: int = np.inf,
             ):

        config = self.af.get_tuned_config(self.scenario,
                                          wallclock_limit=wallclock_limit,
                                          runcount_limit=runcount_limit,
                                          autofolio_config={},
                                          seed=self.seed)
        self.logger.info("Optimized Configuration: %s" % (config))
        return config

    def cross_validation(self, config: Configuration):
        """ run a cross validation on given AutoFolio configuration"""
        score, cv_stat = self.af.run_cv(config=config, scenario=self.scenario, folds=int(self.scenario.cv_data.max().max()))
        self.logger.info("AF's final performance %f" % (score))

        return -1*score, cv_stat

    @staticmethod
    def load_model(load_fn: str):
        with open(load_fn, "br") as fp:
             scenario, feature_pre_pipeline, pre_solver, selector, config = pickle.load(fp)
        facade = AFCsvFacade(scenario=scenario)
        facade.unpickeld = (scenario, feature_pre_pipeline, pre_solver, selector, config)
        return facade

    def predict_instance(self, vec:list):
        af = AutoFolio(random_seed=42)  # random seed doesn't matter here
        if self.unpickeld is None:
            raise ValueError
        pre_pred, pred = af.predict_instance(vec, *self.unpickeld)
        return pre_pred, pred

    def predict_pre(self):
        af = AutoFolio(random_seed=42)  # random seed doesn't matter here
        if self.unpickeld is None:
            raise ValueError
        pre_pred = af.predict_pre(*self.unpickeld)
        return pre_pred

    #fix by @lteu and @felixvuo
    @staticmethod
    def load_and_predict(vec: list,
                         load_fn: str):
        """ get predicted algorithm for given meta-feature vector"""
        af = AutoFolio(random_seed=42)  # random seed doesn't matter here
        pre_pred, pred = af.read_model_and_predict(model_fn=load_fn, feature_vec=vec)
        print("Selected Schedule [(algorithm, budget)]: pre-schedule: %s,        schedule: %s" % (pre_pred, pred))
        return (pre_pred, pred)
