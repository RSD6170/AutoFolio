import logging
import sys
from enum import Flag, auto

from autofolio.aslib.aslib_scenario import ASlibScenario

__author__ = "Marius Lindauer"
__license__ = "BSD"


class Stats(object):

    def __init__(self, runtime_cutoff):
        ''' Constructor

            Arguments
            ---------
            runtime_cutoff: int
                maximal running time
        '''
        self.par1 = 0.0
        self.par10 = 0.0
        self.timeouts = 0
        self.solved = 0
        self.presolve_schedule_solved = 0
        self.unsolvable = 0
        self.presolved_feats = 0
        self.oracle = 0
        self.sbs = 0
        self.reached_oracle = 0

        self.runtime_cutoff = runtime_cutoff

        self.selection_freq = {}
        self.preselection_freq = {}

        self.logger = logging.getLogger("Stats")

    # fix by @kvdblom
    def show(self, remove_unsolvable: bool = True):
        '''
            shows statistics

            Arguments
            --------
            remove_unsolvable : bool
                remove unsolvable from stats

            Returns
            -------
            par10: int
                penalized average runtime
        '''

        if remove_unsolvable and self.runtime_cutoff:
            rm_string = "removed"
            self.logger.debug("Statistics before removing unsolvable instances")
            self.logger.debug("PAR1: %.4f" % (self.par1 / (self.timeouts + self.solved)))
            self.logger.debug("PAR10: %.4f" % (self.par10 / (self.timeouts + self.solved)))
            self.logger.debug("Timeouts: %d / %d" % (self.timeouts, self.timeouts + self.solved))
            timeouts = self.timeouts - self.unsolvable
            par1 = self.par1 - (self.unsolvable * self.runtime_cutoff)
            par10 = self.par10 - (self.unsolvable * self.runtime_cutoff * 10)
            oracle = self.oracle - (self.unsolvable * self.runtime_cutoff * 10)
            sbs = self.sbs - (self.unsolvable * self.runtime_cutoff * 10)
        else:
            rm_string = "not removed"
            timeouts = self.timeouts
            par1 = self.par1
            par10 = self.par10
            oracle = self.oracle
            sbs = self.sbs

        if self.runtime_cutoff:
            n_samples = timeouts + self.solved
            self.logger.debug("n_samples = timeouts + self.solved: %d" % n_samples)
            if n_samples == 0:
                par1_out = sys.maxsize
                par10_out = sys.maxsize
            else:
                par1_out = par1 / n_samples
                par10_out = par10 / n_samples
            self.logger.info("PAR1: %.4f" % (par1_out))
            self.logger.info("PAR10: %.4f" % (par10_out))
            self.logger.info("Timeouts: %d / %d" % (timeouts, n_samples))
            self.logger.info("Presolved during feature computation: %d / %d" % (self.presolved_feats, n_samples))
            self.logger.info("Solved: %d / %d" % (self.solved, n_samples))
            self.logger.info("Solved in pre-schedule: %d / %d" %(self.presolve_schedule_solved, self.solved))
            self.logger.info("As good as oracle: %d / %d", self.reached_oracle, self.solved)
            self.logger.info("Unsolvable (%s): %d / %d" %
                             (rm_string, self.unsolvable, n_samples + self.unsolvable))
        else:
            n_samples = self.solved
            self.logger.info("Number of instances: %d" % (n_samples))
            self.logger.info("Average Solution Quality: %.4f" % (par1 / n_samples))
            par10 = par1
            par10_out = par10 / n_samples

        if n_samples == 0:
            oracle_out = sys.maxsize
        else:
            oracle_out = oracle / n_samples
        self.logger.info("Oracle: %.4f" % (oracle_out))
        if sbs > 0:
            self.logger.info("Single Best: %.4f" % (sbs / n_samples))
        if (sbs - oracle) > 0:
            self.logger.info("Normalized Score: %.4f" % ((par10 - oracle) / (sbs - oracle)))

        self.logger.debug("Selection Frequency")
        for algo, n in self.selection_freq.items():
            if (timeouts + self.solved) == 0:
                frequency = 0
            else:
                frequency = n / (timeouts + self.solved)
            self.logger.debug("%s: %.2f" % (algo, frequency))

        self.logger.debug("Preselection Frequency")
        for algo, n in self.preselection_freq.items():
            if (timeouts + self.solved) == 0:
                frequency = 0
            else:
                frequency = n / (timeouts + self.solved)
            self.logger.debug("%s: %.2f" % (algo, frequency))

        return par10_out

    def merge(self, stat):
        '''
            adds stats from another given Stats objects

            Arguments
            ---------
            stat : Stats
        '''
        self.par1 += stat.par1
        self.par10 += stat.par10
        self.timeouts += stat.timeouts
        self.solved += stat.solved
        self.presolve_schedule_solved += stat.presolve_schedule_solved
        self.unsolvable += stat.unsolvable
        self.presolved_feats += stat.presolved_feats
        self.oracle += stat.oracle
        self.sbs += stat.sbs
        self.reached_oracle += stat.reached_oracle

        for algo, n in stat.selection_freq.items():
            self.selection_freq[algo] = self.selection_freq.get(algo, 0) + n
        for algo, n in stat.preselection_freq.items():
            self.preselection_freq[algo] = self.preselection_freq.get(algo, 0) + n

class StatusEnum(Flag):
    Solved = auto()
    PreSolved = auto()
    PreScheduleSolved = auto()
    Timeouted = auto()


class Validator(object):

    def __init__(self):
        ''' Constructor '''
        self.logger = logging.getLogger("Validation")

    def validate_runtime(self, schedules: dict, test_scenario: ASlibScenario,
                         train_scenario: ASlibScenario = None):
        '''
            validate selected schedules on test instances for runtime

            Arguments
            ---------
            schedules: dict {instance name -> tuples [algo, bugdet]}
                algorithm schedules per instance
            test_scenario: ASlibScenario
                ASlib scenario with test instances
            train_scnenario: ASlibScenario
                ASlib scenario with training instances;
                required for SBS score computation
        '''
        if test_scenario.performance_type[0] != "runtime":
            raise ValueError("Cannot validate non-runtime scenario with runtime validation method")

        stat = Stats(runtime_cutoff=test_scenario.algorithm_cutoff_time)

        feature_times = False
        if test_scenario.feature_cost_data is not None and test_scenario.performance_type[0] == "runtime":
            f_times = test_scenario.feature_cost_data[
                test_scenario.used_feature_groups].sum(axis=1)
            feature_times = True

        feature_stati = test_scenario.feature_runstatus_data[
            test_scenario.used_feature_groups]

        stat.oracle = test_scenario.performance_data.min(axis=1).sum()
        if train_scenario:
            sbs = train_scenario.performance_data.sum(axis=0).idxmin()
            stat.sbs = test_scenario.performance_data.sum(axis=0)[sbs]

        ok_status = test_scenario.runstatus_data == "ok"
        unsolvable = ok_status.sum(axis=1) == 0
        stat.unsolvable += unsolvable.sum()

        for inst, (pre_schedule, schedule) in schedules.items():
            self.logger.debug("Validate %s on %s" % (schedule, inst))


            (schedule_status, schedule_time) = self.evaluate_schedule(f_times, feature_stati, feature_times, inst, schedule, stat, test_scenario)
            (pre_status, pre_time) = self.evaluate_preschedule(inst, pre_schedule, stat, test_scenario)

            if bool( pre_status & StatusEnum.Solved):
                stat.par1 += min(schedule_time, pre_time)
                if(pre_time < schedule_time):
                    stat.presolve_schedule_solved += 1
            else:
                stat.par1 += schedule_time

            if bool(schedule_status & StatusEnum.Solved) and test_scenario.performance_data.idxmin(axis="columns")[inst] == schedule[0][0]:
                stat.reached_oracle += 1

            if bool( (schedule_status | pre_status) & StatusEnum.Solved):
                stat.solved += 1
            if bool( (schedule_status | pre_status) & StatusEnum.PreSolved):
                stat.presolved_feats += 1
            if bool( (schedule_status & pre_status) & StatusEnum.Timeouted):
                stat.timeouts += 1

        stat.par10 = stat.par1 + 9 * \
                     test_scenario.algorithm_cutoff_time * stat.timeouts

        stat.show()

        return stat

    def evaluate_preschedule(self, inst, pre_schedule, stat, test_scenario):
        used_time = 0
        for algo, budget in pre_schedule:
            stat.preselection_freq[algo] = stat.preselection_freq.get(algo, 0) + 1
            time = test_scenario.performance_data[algo][inst]
            used_time += min(time, budget)
            if time <= budget and used_time <= test_scenario.algorithm_cutoff_time and \
                    test_scenario.runstatus_data[algo][inst] == "ok":
                self.logger.debug("Solved by %s (budget: %f -- required to solve: %f)" % (algo, budget, time))
                return StatusEnum.Solved, used_time

            if used_time >= test_scenario.algorithm_cutoff_time:
                self.logger.debug("Timeout after %d" % (used_time))
                return StatusEnum.Timeouted, test_scenario.algorithm_cutoff_time
        return StatusEnum.Timeouted, used_time

    def evaluate_schedule(self, f_times, feature_stati, feature_times, inst, schedule, stat, test_scenario):
        used_time = 0
        if feature_times:
            used_time += f_times[inst]
            self.logger.debug("Used Feature time: %f" % (used_time))

        presolved = False
        for fg in test_scenario.used_feature_groups:
            if "presolved" in feature_stati[fg][inst]:
                presolved = True
                break


        if presolved and used_time < test_scenario.algorithm_cutoff_time:
            self.logger.debug("Presolved during feature computation")
            return StatusEnum.Solved | StatusEnum.PreSolved, used_time

        elif presolved and used_time >= test_scenario.algorithm_cutoff_time:
            return StatusEnum.Timeouted, test_scenario.algorithm_cutoff_time

        else:
            for algo, budget in schedule:
                stat.selection_freq[algo] = stat.selection_freq.get(algo, 0) + 1
                time = test_scenario.performance_data[algo][inst]
                used_time += min(time, budget)
                if time <= budget and used_time <= test_scenario.algorithm_cutoff_time and \
                        test_scenario.runstatus_data[algo][inst] == "ok":
                    self.logger.debug("Solved by %s (budget: %f -- required to solve: %f)" % (algo, budget, time))
                    return StatusEnum.Solved, used_time

                if used_time >= test_scenario.algorithm_cutoff_time:
                    self.logger.debug("Timeout after %d" % (used_time))
                    return StatusEnum.Timeouted, test_scenario.algorithm_cutoff_time
            return None, used_time

    def validate_quality(self, schedules: dict, test_scenario: ASlibScenario,
                         train_scenario: ASlibScenario = None):
        '''
            validate selected schedules on test instances for solution quality

            Arguments
            ---------
            schedules: dict {instance name -> tuples [algo, bugdet]}
                algorithm schedules per instance
            test_scenario: ASlibScenario
                ASlib scenario with test instances
            train_scnenario: ASlibScenario
                ASlib scenario with training instances;
                required for SBS score computation
        '''
        if test_scenario.performance_type[0] != "solution_quality":
            raise ValueError("Cannot validate non-solution_quality scenario with solution_quality validation method")

        self.logger.debug("FYI: Feature costs and algorithm runstatus is ignored")

        stat = Stats(runtime_cutoff=None)

        stat.oracle = test_scenario.performance_data.min(axis=1).sum()
        if train_scenario:
            sbs = train_scenario.performance_data.sum(axis=0).idxmin()
            stat.sbs = test_scenario.performance_data.sum(axis=0)[sbs]

        if test_scenario.maximize[0]:
            test_scenario.performance_data *= -1
            self.logger.debug("Removing *-1 in performance data because of maximization")
            stat.sbs *= -1
            stat.oracle *= -1

        for inst, schedule in schedules.items():
            if len(schedule) > 1:
                self.logger.error("AutoFolio does not support schedules for solution quality")
                sys.exit(9)

            selected_algo = schedule[0][0]
            stat.selection_freq[selected_algo] = stat.selection_freq.get(selected_algo, 0) + 1
            perf = test_scenario.performance_data[selected_algo][inst]

            self.logger.debug("Using %s on %s with performance %f" % (selected_algo, inst, perf))

            stat.par1 += perf
            stat.solved += 1

        stat.show(remove_unsolvable=False)

        return stat
