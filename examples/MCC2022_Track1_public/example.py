import random

import numpy as np

from autofolio.facade.af_csv_facade import AFCsvFacade

#path to data
scenario_path = "examples/MCC2022_Track1_public"

# will be created (or overwritten) by AutoFolio
model_fn = "af_model.pkl"

af = AFCsvFacade(scenario_path=scenario_path, maximize=False)

# fit AutoFolio; will use default hyperparameters of AutoFolio
af.fit()

# tune AutoFolio's hyperparameter configuration for 20 iterations
config = af.tune(runcount_limit=2000)

# evaluate configuration using a 10-fold cross validation
score, _ = af.cross_validation(config=config)

# # re-fit AutoFolio using the (hopefully) better configuration
# # and save model to disk
# af.fit(config=config, save_fn=model_fn)
#
# # feature TT7F-33-24A.cnf
# # best solver: Relaxed_LCMDCBDL_newTech+default
# feature_vec = [3.985074627,0.030427110,3.000000000,4.000000000,0.077570038,0.355700326,0.426080206,5984.000000000,577080.000000000,3684.000000000,577080.000000000,0.624321390,0.000000000,0.006383864,0.993505233,0.076141546,0.000000000,1.000000000,0.049799361,0.008411295,0.144330626,0.000542888,0.009500543,1.237671694,0.000000000,0.008608858,0.013537118,0.008411295,0.642023725,0.000067582,0.028732585,2.148193883,0.932918016,0.243947538,0.000000000,0.999638140,0.310514999,0.000007349,2.031704303,0.000003466,0.000107437,0.296470572,0.011894365,0.003203094,0.391457043,0.000081445,0.005950648,-1.000000000,-0.000000000,-1.000000000,0.000000000,-512.000000000,-1.000000000,-0.000000000,-1.000000000,0.000000000,-0.000000000,0.000000000,0.000000000,0.000000000,0.000542888,0.002442997,0.935315653,0.253110112,0.000000000,0.999765724,0.999727817,0.996660150,0.999695188,0.999634505,0.999684621,0.392734609,0.125695865,0.348119576,0.500000000,0.451733687,0.355986027,0.451451117,0.356054191,0.363864492,156.875000000,0.222975813,867.125000000,0.404507001,989.500000000,0.000000000,997.000000000,1.188385999,0.075126772,0.977996198,0.083463204,134.800000000,0.378679953,782.200000000,0.559617045,985.000000000,0.000000000,990.000000000,0.422024963,1.282210338,0.970942374,0.111052453,None,None,None,None,None,None]
# standard_deviation = 2
#
# mod_feature_vec = [None if x is None else random.normalvariate(0, standard_deviation)*x+x for x in feature_vec]
#
# # load AutoFolio model and
# # get predictions for new meta-feature vector
# pred = AFCsvFacade.load_and_predict(vec=feature_vec, load_fn=model_fn)
#
# pred_mod = AFCsvFacade.load_and_predict(vec=mod_feature_vec, load_fn=model_fn)
#
# print("------------------------")
# print("Unmodified vector:")
# print(pred)
# print("---------------- \nModified vector:")
# print(mod_feature_vec)
# print(pred_mod)
