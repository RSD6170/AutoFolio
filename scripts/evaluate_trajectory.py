import json
import logging
import sys
from collections import Counter

import ConfigSpace
import numpy as np
from matplotlib import pyplot as plt

from autofolio.facade.af_csv_facade import AFCsvFacade


path_intensifier = "/tmp/pycharm_project_718/smac3_output/32bb84e5717c340d5548d8c457cb5dad/12345/intensifier.json"
path_runhistory = "/tmp/pycharm_project_718/smac3_output/32bb84e5717c340d5548d8c457cb5dad/12345/runhistory.json"
scenario_path = "/tmp/pycharm_project_718/examples/test_sharpSAT"
output_path = "out.json"

af = AFCsvFacade(scenario_path=scenario_path, maximize=False)

logging.getLogger("Stats").setLevel(logging.INFO)

af.fit()

values = []
config_analysis = {}
with (open(path_intensifier, "r") as intensifier_file, open(path_runhistory, "r") as runhistory_file, open(output_path, "w") as output):
    intensifier = json.load(intensifier_file)
    runhistory = json.load(runhistory_file)



    incumbent_history = intensifier["trajectory"]

    for incumbent in incumbent_history:
        id = incumbent["config_ids"][0]
        print("Evaluate {}".format(id))
        config_dict = runhistory["configs"][str(id)]

        # ----------------------------------------

        for config_key, config_value in config_dict.items():
            counter = config_analysis.setdefault(config_key, Counter())
            counter[config_value] += 1


        # ----------------------------------------

        config = ConfigSpace.Configuration(af.af.cs, config_dict)
        par10, stat = af.cross_validation(config=config)
        stat_dict = {}
        if stat is not None:
            timeouts = stat.timeouts - stat.unsolvable
            par1 = stat.par1 - (stat.unsolvable * stat.runtime_cutoff)
            par10 = stat.par10 - (stat.unsolvable * stat.runtime_cutoff * 10)
            oracle = stat.oracle - (stat.unsolvable * stat.runtime_cutoff * 10)
            sbs = stat.sbs - (stat.unsolvable * stat.runtime_cutoff * 10)
            n_samples = timeouts + stat.solved
            if n_samples == 0:
                par1_out = sys.maxsize
                par10_out = sys.maxsize
            else:
                par1_out = par1 / n_samples
                par10_out = par10 / n_samples
            stat_dict["par1"]=par1_out
            stat_dict["par10"]=par10_out
            stat_dict["timeout"]=int(timeouts)
            stat_dict["presolvedFeatureComp"]=stat.presolved_feats
            stat_dict["solved"]=stat.solved
            stat_dict["solvedPreschedule"]=stat.presolve_schedule_solved
            stat_dict["asGoodAsOracle"]=stat.reached_oracle
            stat_dict["unsolvable"]=int(stat.unsolvable)

            oracle_out = oracle / n_samples
            stat_dict["oracle"]=oracle_out

            if sbs > 0:
                stat_dict["singleBestSolver"]= (sbs / n_samples)
            if (sbs - oracle) > 0:
                stat_dict["normalizedScore"]= (par10 - oracle) / (sbs - oracle)

            freq = {}
            for algo, n in stat.selection_freq.items():
                if (stat.timeouts + stat.solved) == 0:
                    frequency = 0
                else:
                    frequency = n / (stat.timeouts + stat.solved)
                freq[algo]=frequency
            stat_dict["selectionFrequency"]=freq
            freq = {}
            for algo, n in stat.preselection_freq.items():
                if (stat.timeouts + stat.solved) == 0:
                    frequency = 0
                else:
                    frequency = n / (stat.timeouts + stat.solved)
                freq[algo]=frequency
            stat_dict["preselectionFrequency"]=freq

        values.append({"config_id": id, "trial_id": incumbent["trial"], "walltime": incumbent["walltime"], "config":config_dict, "source":runhistory["config_origins"][str(id)], "stats":stat_dict})

    json.dump({"values":values, "analysis" : config_analysis}, output, indent=4)
