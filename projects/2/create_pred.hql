CREATE MANAGED TABLE hw2_pred (
        id    int,
	pred  int
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION SergeyMalashenko_hw2_pred
;
