import json
import sys

import numpy as np

from autofolio.facade.af_csv_facade import AFCsvFacade

facade: AFCsvFacade

def loadModel(message):
    global facade
    facade = AFCsvFacade.load_model(message["modelPath"])
    answer({"type": "MODEL_LOADED", "modelPath":message["modelPath"]})

def getFeatureGroups(message):
    try:
        scenario, _, _, _, config = facade.unpickeld
        cutoff = scenario.features_cutoff_time

        fgroups = [a for (a,b) in config.items() if a.startswith("fgroup_") and b]

        answer({"type":"FEATURE_GROUPS", "cutoff":cutoff, "fgroups":fgroups})
    except ValueError:
        answer(generateError("Model not loaded!"))

def getPreSchedule(message):
    try:
        pre_sched = facade.predict_pre()
        answer({"type":"PRE_SCHEDULE", "preSchedule":transformTupleList(pre_sched)})
    except ValueError:
        answer(generateError("Model not loaded!"))


def getPrediction(message):
    try:
        pre_pred, pred = facade.predict_instance(handleFeatureVector(message["featureVector"]))
        answer({"type": "PREDICTION", "preSchedule":transformTupleList(pre_pred), "prediction": transformTupleList(pred)})
    except ValueError:
        answer(generateError("Model not loaded!"))

def generateConfig(message):
    temp_facade = AFCsvFacade(scenario_path=message["scenarioPath"], maximize=message["isMaximize"])
    time = message["timeLimit"] if message["timeLimit"] is not None else np.inf
    run = message["runLimit"] if message["runLimit"] is not None else np.inf
    config = temp_facade.tune(wallclock_limit=time, runcount_limit=run)
    answer({"type":"CONFIG", "config":transformConfig(config), "scenarioPath":message["scenarioPath"]})

def generateModel(message):
    global facade
    facade = AFCsvFacade(scenario_path=message["scenarioPath"], maximize=message["maximize"])
    if message["modelPath"] is not None:
        facade.fit(config=getConfig(message["config"]), save_fn=message["modelPath"])
    else:
        facade.fit(config=getConfig(message["config"]))
    answer({"type":"MODEL", "modelPath":message["modelPath"], "scenarioPath":message["scenarioPath"]})


def generateCrossEval(message):
    _, stat = facade.cross_validation(config=getConfig(message["config"]))
    answer({"type":"CROSS_EVALUATION", "config":message["config"], "stats":evaluateStat(stat)})

def handleInput(message):
    try:
        match message["type"]:
            case "LOAD_MODEL":
                loadModel(message)
            case "GET_FEATURE_GROUPS":
                getFeatureGroups(message)
            case "GET_PRE_SCHEDULE":
                getPreSchedule(message)
            case "GET_PREDICTION":
                getPrediction(message)
            case "GENERATE_CONFIG":
                generateConfig(message)
            case "GENERATE_MODEL":
                generateModel(message)
            case "GENERATE_CROSS_EVALUATION":
                generateCrossEval(message)
            case "ERROR":
                print("Error")
            case _:
                answer(generateError("Unknown message type!"))
    except KeyError:
        answer(generateError("Non-valid message format!"))
    except AttributeError:
        answer(generateError("Facade not yet initialized!"))


def answer(message):
    try:
        print(json.dumps(message))
    except TypeError:
        answer(generateError("Error in Json Generation"))


def generateError(reason):
    return {"type":"ERROR", "reason":reason}

def transformTupleList(tuples):
    return [{"solver":s, "budget":t} for (s,t) in tuples]

def transformConfig(config):
    return json.dumps(config)

def getConfig(configString):
    return json.loads(configString)

def handleFeatureVector(messagePart):
    return [getFloatOrNull(x) for x in messagePart.split(",")]

def getFloatOrNull(string):
    try:
        return float(string)
    except ValueError:
        return None

def evaluateStat(stat):
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
        stat_dict["par1"] = par1_out
        stat_dict["par10"] = par10_out
        stat_dict["timeout"] = int(timeouts)
        stat_dict["presolvedFeatureComp"] = stat.presolved_feats
        stat_dict["solved"] = stat.solved
        stat_dict["solvedPreschedule"] = stat.presolve_schedule_solved
        stat_dict["asGoodAsOracle"] = stat.reached_oracle
        stat_dict["unsolvable"] = int(stat.unsolvable)

        oracle_out = oracle / n_samples
        stat_dict["oracle"] = oracle_out

        if sbs > 0:
            stat_dict["singleBestSolver"] = (sbs / n_samples)
        if (sbs - oracle) > 0:
            stat_dict["normalizedScore"] = (par10 - oracle) / (sbs - oracle)

        freq = {}
        for algo, n in stat.selection_freq.items():
            if (stat.timeouts + stat.solved) == 0:
                frequency = 0
            else:
                frequency = n / (stat.timeouts + stat.solved)
            freq[algo] = frequency
        stat_dict["selectionFrequency"] = freq
        freq = {}
        for algo, n in stat.preselection_freq.items():
            if (stat.timeouts + stat.solved) == 0:
                frequency = 0
            else:
                frequency = n / (stat.timeouts + stat.solved)
            freq[algo] = frequency
        stat_dict["preselectionFrequency"] = freq
    return stat_dict

for oline in sys.stdin:
    line = oline.rstrip()
    try:
        handleInput(json.loads(line))
    except json.JSONDecodeError:
        answer(generateError("DecodeError"))

