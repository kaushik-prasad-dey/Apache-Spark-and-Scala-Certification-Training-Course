import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml._
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.streaming.flume._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.OneHotEncoder

object BicycleStreaming {
  case class Bicycle(
  datetime: String, 
  season: Int, 
  holiday: Int, 
  workingday: Int, 
  weather: Int, 
  temp: Double, 
  atemp: Double, 
  humidity: Int, 
  windspeed: Double)
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("byCycleFlumeAgentConfig")
    val sc = new SparkContext(sparkConf)
    val ssc = new StreamingContext(sc, Seconds(2))
    sc.setLogLevel("ERROR")
    val spark = new org.apache.spark.sql.SQLContext(sc)
    import spark.implicits._
    val flumeStream = FlumeUtils.createPollingStream(ssc, "ip-10-1-1-204.ap-south-1.compute.internal", 9092)
    println("Loading tained model.............")    
    val gbt_model = PipelineModel.read.load("/user/kaushikdey45edu/modelSaveOne/")
    val lines = flumeStream.map(event => new String(event.event.getBody().array(), "UTF-8"))
    lines.foreachRDD { rdd => 
      def row(line: List[String]): Bicycle = Bicycle(line(0), line(1).toInt, line(2).toInt,
              line(3).toInt, line(4).toInt, line(5).toDouble, line(6).toDouble, line(7).toInt,
              line(8).toDouble
              )
			  
      val rows_rdd = rdd.map(_.split(",").to[List]).map(row)
      val rows_df = rows_rdd.toDF
    
      if(rows_df.count > 0) {   
        val df_time = rows_df.withColumn("datetime",to_timestamp(col("datetime"),"d-M-y H:m"))
        val datetime_testDF = df_time.
        withColumn("year", year(col("datetime"))).
        withColumn("month", month(col("datetime"))).
        withColumn("day", dayofmonth(col("datetime"))).
        withColumn("hour", hour(col("datetime"))).
        withColumn("minute",minute(col("datetime")))
        //Onehot encoding on season nd weather column
        val indexer = Array("season","weather").map(c => new OneHotEncoder().setInputCol(c).setOutputCol(c + "_Vec"))
        val pipeline = new Pipeline().setStages(indexer)
        val df_r = pipeline.fit(datetime_testDF).transform(datetime_testDF)
        println("Making predictions...............")
        val predictions =  gbt_model.transform(df_r).select($"datetime",$"prediction".as("count"))
        println("Persisting the result to RDBMS..................")
        predictions.write.format("jdbc").
          option("url", "jdbc:jdbc:mysql://ip-10-1-1-204.ap-south-1.compute.internal/kaushikdey45edu").
          option("driver", "com.mysql.cj.jdbc.Driver").option("dbtable", "predictionsModelNew").
          option("user", "kaushikdey45edu").
          option("password", "PurpleCrow52@").
          mode(SaveMode.Append).save
      }
    }
    ssc.start()
    ssc.awaitTermination()   
  }
}
