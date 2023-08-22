import logging
import pickle

from autofolio.facade.af_csv_facade import AFCsvFacade

#path to data
scenario_path = "examples/SAT20-Main/data"

# will be created (or overwritten) by AutoFolio
model_fn = "af_model.pkl"

af = AFCsvFacade(scenario_path=scenario_path, maximize=False)

logging.getLogger("Stats").setLevel(logging.INFO)

with open(model_fn, "br") as fp:
    scenario, feature_pre_pipeline, pre_solver, selector, config = pickle.load(fp)

af.cross_validation(config=config)
