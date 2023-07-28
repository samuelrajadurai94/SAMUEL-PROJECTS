# SAMUEL-PROJECTS
SUPER MARKET SALES PREDICTION:
The dataset contains daily historical sales data for last 3years. The task is to forecast the total amount of products sold in every shop for the test dataset. 
I did outlier removal, data cleaning and converted daily sales data into monthly sales data by using pandas groupby and aggregation function. 
I builded a new data set with all the possible combinations of [‘date_block_num’, ‘shop_id’, ‘item_id’] so we won’t have any missing records. 
I computed Lag features for some important fields. 
I trained dataset with various ML models like lasso regression, Ridge regression, Decisiontree regressor, Randomforest regressor and XGB regressor. 
I did hyperparameter tuning on XGB model. Best model among all is XGB regressor which scored the least rmse value of  0.8539 on validation dataset.
