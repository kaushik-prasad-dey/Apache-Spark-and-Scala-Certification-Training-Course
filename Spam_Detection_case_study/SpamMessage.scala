import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
object SpamMessage {
 def main(args: Array[String]) {
 val spark = SparkSession.builder.appName("Spam Messages Detection").getOrCreate()
 spark.sparkContext.setLogLevel("ERROR")
 // Load data in spark
 val raw =spark.read.option("delimiter","\t").csv("/user/kaushikdey45edu/SpamMessage/SMSSpamCollection").toDF("spam","message")
 // Extract words from the SMS message
 val tokenizer = new Tokenizer().setInputCol("message").setOutputCol("words")
 // Modify the stop words to include your custom words such as â-â
 val stopwords = new StopWordsRemover().getStopWords ++ Array("-")
 val remover = new StopWordsRemover().setStopWords(stopwords).setInputCol("words").setOutputCol("filtered")
 // Create the features from SMS message using CountVectorizer
 val cvmodel = new CountVectorizer().setInputCol("filtered").setOutputCol("features")
 val indexer = new StringIndexer().setInputCol("spam").setOutputCol("label")
 val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
 val pipeline = new Pipeline().setStages(Array(tokenizer, remover, cvmodel,indexer, lr))
 val model = pipeline.fit(raw)
 model.save("/user/kaushikdey45edu/SpamMessage/ModelSave/spam_message.model")
 val output = model.transform(raw)
 println("--->", output.show())
 spark.stop()
 }
}