#!/bin/sh
PYTHONWARNINGS=ignore python3 -c "import exp_benchmark as E; E.run(['Spambase'], n_rep=5)" > results_spambase.log 2>&1
PYTHONWARNINGS=ignore python3 exp_spam_hhnorm.py > results_spam_hhnorm.log 2>&1
echo DONE > results_spam.done
