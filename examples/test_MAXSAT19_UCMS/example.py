import random

import numpy as np

from autofolio.facade.af_csv_facade import AFCsvFacade

#path to data
scenario_path = "examples/test_MAXSAT19_UCMS/data"

# will be created (or overwritten) by AutoFolio
model_fn = "af_model.pkl"

af = AFCsvFacade(scenario_path=scenario_path)

# fit AutoFolio; will use default hyperparameters of AutoFolio
af.fit()

# tune AutoFolio's hyperparameter configuration for 20 iterations
config = af.tune(runcount_limit=25)

# evaluate configuration using a 10-fold cross validation
score = af.cross_validation(config=config)

# re-fit AutoFolio using the (hopefully) better configuration
# and save model to disk
af.fit(config=config, save_fn=model_fn)

# feature close_solutions/teams16_l6a.cnf.wcnf
# best solver: UWrMaxSAT
feature_vec = [2969,66540,2970,66539,-0.000337,0.000015,0.83,0.044635,0.384697,0.407561,0,1,1.025312,0.001204,0.364561,0.000673,0.005387,1.051858,0,0.043072,0.620914,0.05,0.001204,14.667463,0.00003,0.962368,1.655346,0.758216,0.327084,0,1,1.44021,0.001002,15.617237,0.00003,0.853019,1.775754,0.885015,0.000634,1.694881,0.00003,0.04462,0.13,-1,0,-1,0,-512,-1,0,-1,0,0,20.05]
standard_deviation = 2

mod_feature_vec = [random.normalvariate(0, standard_deviation)*x+x for x in feature_vec]

# load AutoFolio model and
# get predictions for new meta-feature vector
pred = AFCsvFacade.load_and_predict(vec=feature_vec, load_fn=model_fn)

pred_mod = AFCsvFacade.load_and_predict(vec=mod_feature_vec, load_fn=model_fn)

print("Unmodified vector:")
print(pred)
print("---------------- \nModified vector:")
print(mod_feature_vec)
print(pred_mod)
