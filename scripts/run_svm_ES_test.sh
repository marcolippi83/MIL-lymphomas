PYTHON=python3

# Perform training on dataset A, and test the model on dataset B, for HL, using Support Vector Machines (SVM) in the embedded space (ES) setting for multiple-instance learning
$PYTHON ../code/svm_ES_test.py ../data/dataset_A.csv ../data/dataset_B.csv HL > log_svm_test_ES_HL.txt



