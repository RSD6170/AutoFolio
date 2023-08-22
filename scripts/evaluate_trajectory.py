import json

import ConfigSpace
import numpy as np
from matplotlib import pyplot as plt

from autofolio.facade.af_csv_facade import AFCsvFacade


path_intensifier = "/tmp/pycharm_project_349/smac3_output/f06f4b0a069130b42ce8e8ec32c2cf94/12345/intensifier.json"
path_runhistory = "/tmp/pycharm_project_349/smac3_output/f06f4b0a069130b42ce8e8ec32c2cf94/12345/runhistory.json"
scenario_path = "/tmp/pycharm_project_349/examples/test_sharpSAT"

af = AFCsvFacade(scenario_path=scenario_path, maximize=False)
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
