import random

import numpy as np

from autofolio.facade.af_csv_facade import AFCsvFacade

#path to data
scenario_path = "examples/MCC2022_T1_randomSplits/Training_1"

# will be created (or overwritten) by AutoFolio
model_fn = "af_model.pkl"

af = AFCsvFacade(scenario_path=scenario_path, maximize=False)

# fit AutoFolio; will use default hyperparameters of AutoFolio
af.fit()

# tune AutoFolio's hyperparameter configuration for 20 iterations
config = af.tune(runcount_limit=6000)

# evaluate configuration using a 10-fold cross validation
score, _ = af.cross_validation(config=config)

# # re-fit AutoFolio using the (hopefully) better configuration
# # and save model to disk
# af.fit(config=config, save_fn=model_fn)

