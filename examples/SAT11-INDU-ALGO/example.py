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
config = af.tune(runcount_limit=80)

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
print("---------------------------------------------")


# feature hwmcc10-timeframe-expansion-k50-pdtpmsns2-tseitin
# best solver: CryptoMiniSat_Strange-Night2-st_fixed_
feature_vec = [88352.0,262658.0,23820.0,135563.0,2.7092,0.9375,0.1757,0.456,0.7582,0.0,1.0,1.7048,0.0001,0.3687,0.0001,0.0018,1.4907,0.0,0.2359,0.6345,0.0001,1.193,0.0001,0.0021,3.1253,0.1268,0.1192,0.0,0.8148,2.896,0.0001,1.2783,0.0,0.0007,2.6411,0.4676,0.0001,1.0172,0.0,0.0015,0.0005,0.8597,0.0,0.0036,4.9056,0.1392,0.6055,0.0217,0.7333,3.3076,11.4659,0.0602,9.0,12.0,0.9341,561.4,0.3019,147.0,708.0,708.0,399.0,695.0,490.0,643.0,30.6903,2.2207,6.3182,235.1361,235.1361,7.351,8.7621,7.7433,8.1192,0.3554,0.7191,0.0,1.0,0.7174,0.0222,0.5776,0.1252,0.3393,0.049,0.8863,0.0,0.2859,0.1088,0.0009,0.0778,0.0122,0.039,1378.0,0.8273,6187.67,0.0062,6199.0,6145.0,6199.0,1.1092,0.7873,0.8941,0.0798,1481.73,0.0257,6278.36,0.0097,6286.0,6196.0,6338.0,0.3075,0.2532,0.9022,0.0034,0.0635,0.1791]
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
print("---------------------------------------------")