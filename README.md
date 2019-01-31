# MIL-lymphomas

Repository for the data and source code for the paper "Texture Analysis and Multiple-Instance Learning for the Classification of Malignant Lymphomas", submitted for publication.

The repository contains the following folders:

* code
-- svm_IS_loo.py (leave-one-out instance-space multiple-instance learning with svm)
-- svm_ES_loo.py (leave-one-out embedded-space multiple-instance learning with svm)
-- svm_ES_test.py (training/test split with embedded-space multiple-instance learning with svm)
-- rf_ES_loo.py (leave-one-out embedded-space multiple-instance learning with random forests)

* data
-- dataset_A.csv
-- dataset_B.csv
-- dataset_A+B.csv
-- dataset_A_large.csv (small VOIs removed)
-- dataset_B_large.csv (small VOIs removed)
-- dataset_A+B_large.csv (small VOIs removed)

* scripts (bash scripts to run python code)
-- run_svm_IS_loo.sh
-- run_svm_ES_loo.sh
-- run_svm_ES_test.sh
-- run_rf_ES_loo.sh



