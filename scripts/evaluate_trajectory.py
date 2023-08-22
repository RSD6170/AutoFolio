import json
import logging

import ConfigSpace
import numpy as np
from matplotlib import pyplot as plt

from autofolio.facade.af_csv_facade import AFCsvFacade


path_intensifier = "/home/ubuntu/MCC22/MCC_T1_public_2000I/12345/intensifier.json"
path_runhistory = "/home/ubuntu/MCC22/MCC_T1_public_2000I/12345/runhistory.json"
scenario_path = "/home/ubuntu/AutoFolio/examples/MCC2022_Track1_public"

af = AFCsvFacade(scenario_path=scenario_path, maximize=False)

logging.getLogger("Stats").setLevel(logging.INFO)

af.fit()

values = {}
with (open(path_intensifier, "r") as intensifier_file, open(path_runhistory, "r") as runhistory_file):
    intensifier = json.load(intensifier_file)
    runhistory = json.load(runhistory_file)



    incumbent_history = intensifier["trajectory"]

    for incumbent in incumbent_history:
        id = incumbent["config_ids"][0]
        print("Evaluate {}".format(id))
        config_dict = runhistory["configs"][str(id)]

        config = ConfigSpace.Configuration(af.af.cs, config_dict)
        values[incumbent["trial"]]=-1 * af.cross_validation(config=config)


print(values)
values.pop(1)
fig, ax = plt.subplots()

ax.step(values.keys(), values.values(), linewidth=2.5)


plt.show()
