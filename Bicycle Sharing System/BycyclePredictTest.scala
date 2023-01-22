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

object BicyclePredictTest {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("Bicycle_Sharing_System_Analysis_bar")
    val sc = new SparkContext(sparkConf)
    sc.setLogLevel("ERROR")
    val spark = new org.apache.spark.sql.SQLContext(sc)
    import spark.implicits._
    println("Reading Training data log")
    val testDF = spark.read.format("csv").
           option("inferSchema",true).
           option("header",true).
           load("/user/kaushikdey45edu/BikeSharingApplicationOne/test.csv")
    println("Cleaning data log")
    //Converting datetime string column to timestamp column
    val df_time = testDF.withColumn("datetime", to_timestamp(col("datetime"),"d-M-y H:m"))
    
    //Now Spliting date time into meaning columns such as year,month,day,hour
    val datetime_testDF = df_time.
    withColumn("year", year(col("datetime"))).
    withColumn("month", month(col("datetime"))).
    withColumn("day", dayofmonth(col("datetime"))).
    withColumn("hour", hour(col("datetime"))).
    withColumn("minute",minute(col("datetime")))
    
    //Onehot encoding on season and weather column.
    val indexer = Array("season","weather").map(c=>new OneHotEncoder().setInputCol(c).setOutputCol(c + "_Vec"))
    val pipeline = new Pipeline().setStages(indexer)
    val df_r = pipeline.fit(datetime_testDF).transform(datetime_testDF)
    
    println("Loading Trained Model..................")
    val gbt_model = PipelineModel.read.load("/user/kaushikdey45edu/modelSaveOne/")
    println("Making predictions....................") 
    val predictions = gbt_model.transform(df_r).select($"datetime",$"prediction".as("count"))
    println("Persisting the result to RDBMS................." + predictions)
    predictions.write.format("jdbc").
      option("url", "jdbc:mysql://ip-10-1-1-204.ap-south-1.compute.internal/kaushikdey45edu").
      option("driver", "com.mysql.cj.jdbc.Driver").option("dbtable", "predictionsModel").
      option("user", "kaushikdey45edu").
      option("password", "PurpleCrow52@").
      mode(SaveMode.Append).save
  }
}
