import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.OneHotEncoder

object BicyclePredictModel{
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("Bicycle_Sharing_System_Analysis")
    val sc = new SparkContext(sparkConf)
    sc.setLogLevel("ERROR")
    val spark = new org.apache.spark.sql.SQLContext(sc)
    import spark.implicits._
    println("Reading training data log")
    val trainDF = spark.read.format("csv").
         option("inferSchema",true).
         option("header",true).
         load("/user/kaushikdey45edu/BikeSharingApplicationOne/train.csv")
    println("Cleaning data log")
    //Converting datetime string column to timestamp column
    val df_time = trainDF.withColumn("datetime", to_timestamp(col("datetime"),"d-M-y H:m"))
    //Now Spliting date time into meaning columns such as year,month,day,hour
    val datetime_trainDF = df_time.
    withColumn("year", year(col("datetime"))).
    withColumn("month", month(col("datetime"))).
    withColumn("day", dayofmonth(col("datetime"))).
    withColumn("hour", hour(col("datetime"))).
    withColumn("minute",minute(col("datetime")))   
    //Onehot encoding on season and weather column.
    val indexer = Array("season","weather").map(c=>new OneHotEncoder().setInputCol(c).setOutputCol(c + "_Vec"))
    val pipeline = new Pipeline().setStages(indexer)
    val df_r = pipeline.fit(datetime_trainDF).transform(datetime_trainDF)
    //split data into train test
    val splitSeed =123
    val Array(train, train_test) = df_r.randomSplit(Array(0.7, 0.3), splitSeed)
    //Generate Feature Column
    val feature_cols = Array("holiday","workingday","temp","atemp","humidity","windspeed","season_Vec","weather_Vec","year","month","day","hour","minute")
    //Assemble Feature selection
    val assembler = new VectorAssembler().setInputCols(feature_cols).setOutputCol("features")
    //Model Building
    val gbt = new GBTRegressor().setLabelCol("count").setFeaturesCol("features")
    val pipeline2 = new Pipeline().setStages(Array(assembler,gbt))
    println("Training model.........................")
    val gbt_model = pipeline2.fit(train)
    val predictions = gbt_model.transform(train_test)
    val evaluator = new RegressionEvaluator().setLabelCol("count").setPredictionCol("prediction").setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("GBT Regressor Root Mean Squared Error (RMSE) on train_test data = " + rmse)
    println("Persisting the model................")
    gbt_model.write.overwrite().save("/user/kaushikdey45edu/modelSaveOne/")
   }
}
