import random

import numpy as np

from autofolio.facade.af_csv_facade import AFCsvFacade

#path to data
scenario_path = "examples/SAT11-INDU-ALGO/data"

# will be created (or overwritten) by AutoFolio
model_fn = "af_model.pkl"

af = AFCsvFacade(scenario_path=scenario_path)

# fit AutoFolio; will use default hyperparameters of AutoFolio
af.fit()

# tune AutoFolio's hyperparameter configuration for 20 iterations
config = af.tune(runcount_limit=50)

# evaluate configuration using a 10-fold cross validation
score = af.cross_validation(config=config)

# re-fit AutoFolio using the (hopefully) better configuration
# and save model to disk
af.fit(config=config, save_fn=model_fn)

# feature SAT09/APPLICATIONS/bitverif/maxor/maxor128.cnf
# best solver: Lingeling_587f_fixed_
feature_vec = [200308.0,598619.0,72981.0,341039.0,1.7447,0.7553,0.214,0.4387,0.7789,0.0,1.0,1.3966,0.0,1.0202,0.0,0.0018,1.221,0.0,0.2671,0.826,0.0,2.378,0.0,0.0021,2.1095,0.1641,0.1814,0.0,0.8483,2.2843,0.0,1.6769,0.0,0.0009,1.7529,0.5216,0.0,3.0828,0.0,0.0022,None,None,None,None,None,None,None,None,None,None,6.7455,0.0858,5.0,7.0,0.5965,153.0,0.0,153.0,153.0,153.0,153.0,153.0,153.0,153.0,25.3072,0.0,25.3072,25.3072,25.3072,25.3072,25.3072,25.3072,25.3072,0.2938,0.701,0.0,0.9314,0.5304,0.01,0.5153,0.1464,0.2573,0.0606,0.7679,0.0,0.4592,0.1325,0.0233,0.0977,0.0246,0.0305,26856.0,0.0,0.0,0.0,0.0,0.0,0.0,3.8009,0.0,2.2756,0.0,17552.0,0.0,0.0,0.0,0.0,0.0,0.0,2.7709,0.0,1.5846,0.0,0.2134,0.6036]
standard_deviation = 0.2

mod_feature_vec = [random.normalvariate(0, standard_deviation)*x+x if x is not None else None for x in feature_vec]

# load AutoFolio model and
# get predictions for new meta-feature vector
pred = AFCsvFacade.load_and_predict(vec=feature_vec, load_fn=model_fn)

pred_mod = AFCsvFacade.load_and_predict(vec=mod_feature_vec, load_fn=model_fn)

print("Unmodified vector:")
print(pred)
print("---------------- \nModified vector:")
print(mod_feature_vec)
print(pred_mod)
