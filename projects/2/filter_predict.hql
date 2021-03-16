add file 2.joblib;
add file projects/2/model.py;
add file projects/2/predict.py;

insert overwrite table hw2_pred
select transform(id, if1, if2, if3, if4, if5, if6, if7, if8, if9, if10, if11, if12, if13,
cf1, cf2, cf3, cf4, cf5, cf6, cf7, cf8, cf9, cf10, cf11, cf12, cf13, cf14, cf15, cf16, cf17,
cf18, cf19, cf20, cf21, cf22, cf23, cf24, cf25, cf26, day_number)
using 'predict.py' from hw2_test where if1 > 20 and if1 < 40;
