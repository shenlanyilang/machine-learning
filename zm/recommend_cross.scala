package com.zm

/*import org.apache.log4j
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.apache.log4j.Level*/
import org.apache.spark.sql._
import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.tuning.{CrossValidator,ParamGridBuilder,CrossValidatorModel}
import org.apache.spark.ml.recommendation.{ALS,ALSModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.param.ParamMap
import java.io.PrintWriter

object recommend_cross {

  case class Rating(userId:Int, movieId:Int, rating:Float, timestamp:Long)

  val moviePath = "D:\\spark_local\\ml-100k\\u1.data" // Should be some file on your system
  def parseRating(str: String):Rating = {
    val str2 = str.replaceAll(" +",",")
    val fileds = str2.split(",")
    assert(fileds.size==4)
    Rating(fileds(0).toInt, fileds(1).toInt, fileds(2).toFloat, fileds(3).toLong)
  }

  def rddToDF(sparkSession: SparkSession):DataFrame = {
    import sparkSession.implicits._
    val movieData = sparkSession.sparkContext.textFile("file:\\"+moviePath)
    val rating = movieData.map(parseRating).toDF()
    return rating
  }
  def main(args: Array[String]): Unit ={
    //Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    //log4j.Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    /*val logger:Logger = LoggerFactory.getLogger("log_test")
    logger.info("----------------------------------")
    logger.info("program start...")*/
    val sparkSession = SparkSession.builder()
      .appName("my_test")
      .master("local")
      .getOrCreate()

    val ratings = rddToDF(sparkSession)
    //ratings.rdd.saveAsObjectFile("D:\\spark_local\\ml-100k\\u1_rdd.data")
    val ratings2 = sparkSession.sparkContext.objectFile("D:\\spark_local\\ml-100k\\u1_rdd.data")
    //ratings2.first()
    val cf_model = new ALS().setImplicitPrefs(false).setMaxIter(10).setUserCol("userId")
      .setItemCol("movieId").setRatingCol("rating").setPredictionCol("prediction")
      .setRank(10)
      .setColdStartStrategy("drop")
    println(ratings2.getClass.getName)

    val Array(train,test) = ratings.randomSplit(Array(0.8,0.2))
    val transformTest = train.select("userId","movieId","timestamp","rating")
    val cf_evaluator = new RegressionEvaluator().setMetricName("rmse")
      .setLabelCol("rating").setPredictionCol("prediction")

   /* val cf_evaluator = new MyEvaluator().setMetricName("areaUnderROC")
      .setLabelCol("rating").setRawPredictionCol("prediction")*/

    val pipeline = new Pipeline().setStages(Array(cf_model))
    val paramGrid = new ParamGridBuilder().addGrid(cf_model.alpha,Array(0.5))
      .addGrid(cf_model.regParam,Array(0.01))
      .build()
    val cv = new CrossValidator().setEstimator(pipeline)
      .setEvaluator(cf_evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cv_model = cv.fit(ratings)
    val avg = cv_model.avgMetrics
    avg.foreach(x => println("avg is "+x))
    val best_model = cv_model.bestModel.asInstanceOf[PipelineModel]
    val best_cf = best_model.stages(0).asInstanceOf[ALSModel]
    println(avg.mkString(sep=","))
    implicit class BestParamMapCrossValidatorModel(cvModel: CrossValidatorModel) {
      def bestEstimatorParamMap: ParamMap = {
        cvModel.getEstimatorParamMaps
          .zip(avg)
          .minBy(_._2)
          ._1
      }
    }
    def bestEstimatorParamMap: ParamMap = {
      cv_model.getEstimatorParamMaps
        .zip(cv_model.avgMetrics)
        .maxBy(_._2)
        ._1
    }
    val params = bestEstimatorParamMap
    val s1 = bestEstimatorParamMap.toString()
    val sc = sparkSession.sparkContext
    val writeStr = sc.parallelize(Array(s1))
    writeStr.saveAsTextFile("D:\\spark_local\\ml-100k\\parameter2.txt")
    val file1 = new PrintWriter("D:\\spark_local\\ml-100k\\parameter.txt")
    file1.write(s1)
    file1.close()
    println("s1 type is "+s1.getClass.getName)
    println("best estimator params is "+bestEstimatorParamMap.toString())
    //println("params type is "+bestEstimatorParamMap.getClass.getName)
    val result = best_cf.transform(transformTest)
    result.printSchema()
    result.show(20)
  }
}

