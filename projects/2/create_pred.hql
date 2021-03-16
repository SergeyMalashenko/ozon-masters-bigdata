CREATE MANAGED TABLE hw2_pred (
        id    string,
	pred  double
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION 'SergeyMalashenko_hw2_pred'
;
