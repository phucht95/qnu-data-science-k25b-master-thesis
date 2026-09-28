#!/bin/sh
# after exp_credit.py: refresh the interpretation (grouped importances), stability and monotone analyses,
# then release exp_cost.py (which waits for results_extra.done)
until grep -q "interpretation saved" results_credit.log 2>/dev/null; do sleep 15; done
while pgrep -f "exp_credit.py" > /dev/null; do sleep 5; done
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python3 -c "import exp_credit; exp_credit.interpret()" > results_extra.log 2>&1
python3 credit_stability.py >> results_extra.log 2>&1
python3 exp_monotone.py >> results_extra.log 2>&1
echo DONE > results_extra.done
