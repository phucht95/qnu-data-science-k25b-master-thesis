#!/bin/sh
# wait for the benchmark, then run the remaining experiments one after another
while pgrep -f exp_benchmark.py > /dev/null; do sleep 10; done
PYTHONWARNINGS=ignore python3 exp_sensitivity.py > results_sensitivity.log 2>&1
PYTHONWARNINGS=ignore python3 exp_credit.py > results_credit.log 2>&1
PYTHONWARNINGS=ignore python3 exp_cost.py > results_cost.log 2>&1
echo ALLDONE > results_rest.done
