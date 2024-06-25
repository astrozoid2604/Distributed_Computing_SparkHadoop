from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, substring, when
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import sys
import os
from time import time



def concatenate_output(output_dir, output_file, prefix='part-'):
    os.chdir(output_dir)
    input_files = [file for file in os.listdir() if file.startswith(prefix)]
    with open('../' + output_file, 'w') as output:
        for i, input_file in enumerate(input_files):
            with open(input_file, 'r') as input:
                if i != 0:
                    next(input)  # Skip the header for all but the first file
                output.write(input.read())
    os.chdir('../')


def main(input_directory, output_path):
    # Initialize a Spark session
    spark = SparkSession.builder \
        .appName("Concatenate CSV to DataFrame") \
        .getOrCreate()


    #########################################
    #### Data Cleaning & Transformation #####
    #########################################

    # Check if the external file exists
    if os.path.exists(output_path):
        print(f"\nFile {output_path} already exists. Reading the DataFrame from the file...\n")
        combined_df = spark.read.csv(output_path, header=True, inferSchema=True)
    else:
        start = time()
        temporary_directory = "./" + output_path.split('./')[-1].split('.')[0]  # If output_path='./combined_df.csv', temporary_directory='./combined_df'

        # List all CSV files in the input directory
        csv_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.csv')]

        # Read all CSV files at once
        df = spark.read.csv(csv_files, header=True, inferSchema=True)

        # Define the columns to keep
        columns_to_keep = ['LATITUDE', 'LONGITUDE', 'ELEVATION', 'WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP']

        # Filter the columns and apply the necessary conditions
        df_filtered = df.select(columns_to_keep) \
            .filter(~(split(col('WND'), ',')[0] == '999')) \
            .filter(~(split(col('WND'), ',')[3] == '9999')) \
            .filter(~(split(col('CIG'), ',')[0] == '99999')) \
            .filter(~(split(col('VIS'), ',')[0] == '999999')) \
            .filter(~(split(col('TMP'), ',')[0] == '+9999')) \
            .filter(~(split(col('DEW'), ',')[0] == '+9999')) \
            .filter(~(split(col('SLP'), ',')[0] == '99999'))

        # Transform columns
        combined_df = df_filtered.select(
            col('LATITUDE').cast('float'),
            col('LONGITUDE').cast('float'),
            col('ELEVATION').cast('float'),
            split(col('WND'), ',')[0].alias('WIND_DIRECTION').cast('int'),
            split(col('WND'), ',')[3].alias('WIND_SPEED').cast('int'),
            split(col('CIG'), ',')[0].alias('CEILING_HEIGHT').cast('int'),
            split(col('VIS'), ',')[0].alias('VISIBILITY').cast('int'),
            split(when(col('TMP').startswith('+'), col('TMP').substr(2, 4)).otherwise(col('TMP')), ',')[0].alias('AIR_TEMPERATURE').cast('int'),
            split(when(col('DEW').startswith('+'), col('DEW').substr(2, 4)).otherwise(col('DEW')), ',')[0].alias('DEW_POINT_TEMPERATURE').cast('int'),
            split(col('SLP'), ',')[0].alias('ATMOSPHERIC_PRESSURE_OBSERVATION').cast('int')
        )

        # Save the combined DataFrame to an external file with headers
        combined_df.write.csv(temporary_directory, header=True, mode='overwrite')

        # Concatenate the output files from all slave nodes
        concatenate_output(temporary_directory, output_path)
        print(f"\nCombined DataFrame saved to {output_path} which is finished in {time() - start:.2f} seconds...\n")

    # Print details of the combined_df
    combined_df.show(10)
    combined_df.printSchema()
    num_rows = combined_df.count()
    num_cols = len(combined_df.columns)
    print(f"\nDimensions of the combined DataFrame: {num_rows} rows, {num_cols} columns...\n")


    #########################################
    ########## MinMax Scaling ###############
    #########################################

    feature_columns = ['LATITUDE', 'LONGITUDE', 'ELEVATION', 'WIND_DIRECTION', 'WIND_SPEED', 'CEILING_HEIGHT', 'VISIBILITY', 'AIR_TEMPERATURE', 'DEW_POINT_TEMPERATURE']
    min_values = [-90000, -179999, -400, 1, 0, 0, 0, -932, -982]
    max_values = [90000, 180000, 8850, 360, 900, 22000, 160000, 618, 368]

    # Apply Min-Max scaling manually
    for col_name, min_val, max_val in zip(feature_columns, min_values, max_values):
        combined_df = combined_df.withColumn(col_name, (col(col_name) - min_val) / (max_val - min_val))

    # Show the scaled DataFrame
    combined_df.show(10)
    combined_df.printSchema()


    #########################################
    ######## Train-Test Split ###############
    #########################################

    train_df, test_df = combined_df.randomSplit([0.7, 0.3], seed=42)
    print(f"\nTraining set has {train_df.count()} rows.")
    print(f"Test set has {test_df.count()} rows.\n")


    #########################################
    ######## Model Pipeline #################
    #########################################

    # Assemble features
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Define regression models
    ridge_reg = LinearRegression(featuresCol="features", labelCol="ATMOSPHERIC_PRESSURE_OBSERVATION", elasticNetParam=0.0)
    gbt = GBTRegressor(featuresCol="features", labelCol="ATMOSPHERIC_PRESSURE_OBSERVATION")
    rf = RandomForestRegressor(featuresCol="features", labelCol="ATMOSPHERIC_PRESSURE_OBSERVATION")

    # Create parameter grids for hyperparameter tuning
    ridge_param_grid = (ParamGridBuilder()
                        .addGrid(ridge_reg.regParam, [0.01, 0.1, 1.0])
                        .build())

    gbt_param_grid = (ParamGridBuilder()
                      .addGrid(gbt.maxDepth, [5, 7, 9])
                      .addGrid(gbt.maxIter, [20, 30, 40])
                      .build())

    rf_param_grid = (ParamGridBuilder()
                     .addGrid(rf.numTrees, [20, 30, 40])
                     .addGrid(rf.maxDepth, [5, 7, 9])
                     .build())

    # Define evaluators
    evaluator = RegressionEvaluator(labelCol="ATMOSPHERIC_PRESSURE_OBSERVATION", predictionCol="prediction", metricName="rmse")

    # Create CrossValidator for each model
    ridge_cv = CrossValidator(estimator=ridge_reg, estimatorParamMaps=ridge_param_grid, evaluator=evaluator, numFolds=3)
    gbt_cv = CrossValidator(estimator=gbt, estimatorParamMaps=gbt_param_grid, evaluator=evaluator, numFolds=3)
    rf_cv = CrossValidator(estimator=rf, estimatorParamMaps=rf_param_grid, evaluator=evaluator, numFolds=3)

    # Create pipelines for each model
    ridge_pipeline = Pipeline(stages=[assembler, ridge_cv])
    gbt_pipeline = Pipeline(stages=[assembler, gbt_cv])
    rf_pipeline = Pipeline(stages=[assembler, rf_cv])

    # Fit the models
    ridge_model = ridge_pipeline.fit(train_df)
    gbt_model = gbt_pipeline.fit(train_df)
    rf_model = rf_pipeline.fit(train_df)

    # Make predictions on the train and test sets
    ridge_train_predictions = ridge_model.transform(train_df)
    gbt_train_predictions = gbt_model.transform(train_df)
    rf_train_predictions = rf_model.transform(train_df)

    ridge_test_predictions = ridge_model.transform(test_df)
    gbt_test_predictions = gbt_model.transform(test_df)
    rf_test_predictions = rf_model.transform(test_df)

    # Evaluate the models
    ridge_train_rmse = evaluator.evaluate(ridge_train_predictions)
    gbt_train_rmse = evaluator.evaluate(gbt_train_predictions)
    rf_train_rmse = evaluator.evaluate(rf_train_predictions)

    ridge_test_rmse = evaluator.evaluate(ridge_test_predictions)
    gbt_test_rmse = evaluator.evaluate(gbt_test_predictions)
    rf_test_rmse = evaluator.evaluate(rf_test_predictions)

    print("\n====================================================================")
    print("========================= FINAL OUTPUT =============================")
    print("====================================================================\n")

    print(f"\nRidge Regression Train RMSE: {ridge_train_rmse}")
    print(f"Gradient-Boosted Tree Train RMSE: {gbt_train_rmse}")
    print(f"Random Forest Train RMSE: {rf_train_rmse}\n")

    print(f"\nRidge Regression Test RMSE: {ridge_test_rmse}")
    print(f"Gradient-Boosted Tree Test RMSE: {gbt_test_rmse}")
    print(f"Random Forest Test RMSE: {rf_test_rmse}\n")

    # Print best hyperparameters
    ridge_best_model = ridge_model.stages[-1].bestModel
    gbt_best_model = gbt_model.stages[-1].bestModel
    rf_best_model = rf_model.stages[-1].bestModel
    print("\nBest Hyperparameters:")
    print(f"Ridge Regression: regParam = {ridge_best_model._java_obj.parent().getRegParam()}")
    print(f"GBT: maxDepth = {gbt_best_model._java_obj.parent().getMaxDepth()}, maxIter = {gbt_best_model._java_obj.parent().getMaxIter()}")
    print(f"Random Forest: numTrees = {rf_best_model._java_obj.parent().getNumTrees()}, maxDepth = {rf_best_model._java_obj.parent().getMaxDepth()}\n")

    # Stop the Spark session
    spark.stop()



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <path_to_directory> <output_path>")
        sys.exit(-1)
    
    input_directory = sys.argv[1]
    output_path = sys.argv[2]
    main(input_directory, output_path)
