PYTHON=python3

for lymph in DLBCL HL FL MCL
do

# Run leave-one-out (LOO) validation on dataset A for Support Vector Machines (SVM), for all lymphoma subtypes, using the instance space (IS) paradigm for multiple-instance learning
$PYTHON ../code/svm_IS_loo.py ../data/dataset_A.csv $lymph > log_A_svm_IS_$lymph.txt

done

# Run leave-one-out (LOO) validation on dataset A+B for Support Vector Machines (SVM), for HL, using the instance space (IS) paradigm for multiple-instance learning
$PYTHON ../code/svm_IS_loo.py ../data/dataset_A+B.csv HL > log_A+B_svm_IS_HL.txt

