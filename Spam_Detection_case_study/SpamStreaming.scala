import org.apache.spark.sql.SparkSession
//import org.apache.spark.ml.feature.QuantileDiscretizer
//import org.apache.spark.sql.types._
//import org.apache.spark.ml.feature.VectorAssembler
//import org.apache.spark.ml.linalg.Vectors
//import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
//import org.apache.spark.ml.evaluation.RegressionEvaluator
//import org.apache.spark.ml.regression.{RandomForestRegressionModel,RandomForestRegressor}
import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.sql.Encoders
import org.apache.spark.ml._
case class Message(message:String)
object SpamStreaming {
 def main(args: Array[String]) {
 val conf = new SparkConf().setAppName("SpamStreaming")
 val ssc = new StreamingContext(conf, Seconds(10))
 val lines = ssc.textFileStream("/user/kaushikdey45edu/spam_message/SMSSpamCollection")
 lines.foreachRDD {
     rdd =>val spark=SparkSession.builder().getOrCreate()
	 import spark.implicits._
     val rawRdd = rdd.map(Message(_))
     val raw = spark.createDataFrame(rawRdd)
     val pipeline = PipelineModel.read.load("/user/kaushikdey45edu/SpamMessage/ModelSave/spam_message.model")
     val predictions = pipeline.transform(raw)
     val prop = new java.util.Properties
     prop.put("driver", "com.mysql.jdbc.Driver");
     prop.put("url", "jdbc:jdbc:mysql://ip-10-1-1-204.ap-south-1.compute.internal/kaushikdey45edu");
     prop.put("user", "kaushikdey45edu");
     prop.put("password", "PurpleCrow52@");
     predictions.select("message","prediction").write.mode("append").jdbc(
			prop.getProperty("url"), "spam_message", prop)
	}
 ssc.start()
 ssc.awaitTermination()
 }
 }