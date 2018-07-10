package com.zm

import org.apache.spark.sql._
import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.tuning.{CrossValidator,ParamGridBuilder,CrossValidatorModel}
import org.apache.spark.ml.recommendation.{ALS,ALSModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.param.ParamMap

object recommendCrossValid {
  def main(args: Array[String]): Unit ={

    val sparkSession = SparkSession.builder()
      .appName("newsappRecommendCrossvalid")
      .enableHiveSupport()
      .getOrCreate()

    val regParam =  Array(1.0, 0.5, 0.2, 0.1, 0.05)
    val alpha = Array(2.0)
    val rank = Array(200)

    //测试时参数
    /*val regParam =  Array(0.1)
    val alpha = Array(2.0)
    val rank = Array(200)*/

    val maxIter = 10
    val userCol = "userId"
    val itemCol = "articleId"
    val ratingCol = "rating"
    val predictCol = "prediction"

    sparkSession.sql("use newsapp")
    val trainSamples = sparkSession.sql("select cf_uid userId,cf_article_id articleId,rating,article_time timestamp " +
      "from newsapp.cf_train_sample_filter where day='2018-07-01'")
    val userPreRec = sparkSession.sql("select cf_uid userId,cf_article_id articleId,id,article_id " +
      "from cf_pre_rec_article")
    trainSamples.persist()
    userPreRec.persist()
    val Array(train, test) = trainSamples.randomSplit(Array(0.8,0.2), seed = 20L)
    /*val articleNumbers = trainSamples.dropDuplicates(Array("articleId")).count()
    val userNumbers = trainSamples.dropDuplicates(Array("userId")).count()*/

    val als = new ALS()
      .setImplicitPrefs(false)
      .setMaxIter(maxIter)
      .setUserCol(userCol)
      .setItemCol(itemCol)
      .setRatingCol(ratingCol)
      .setPredictionCol(predictCol)
      .setSeed(20L)
      .setColdStartStrategy("drop")

    val crossValidEvaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(als))
    val paramGrid = new ParamGridBuilder()
      .addGrid(als.alpha,alpha)
      .addGrid(als.regParam,regParam)
      .addGrid(als.rank,rank)
      .build()
    val crossValidModel = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(crossValidEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    println("start training model...")
    val CrossValidModel = crossValidModel.fit(train)
    println("model training completed...")
    val avg = CrossValidModel.avgMetrics
    val bestPipeline = CrossValidModel.bestModel.asInstanceOf[PipelineModel]
    val bestAls = bestPipeline.stages(0).asInstanceOf[ALSModel]
    val bestParamMetric = CrossValidModel
      .getEstimatorParamMaps
      .zip(avg)
        .minBy(_._2)
    val bestParam = bestParamMetric._1
    val bestMetric = bestParamMetric._2
    println(bestParam.toString())
    println(bestMetric.toString())
    val bestParamPath = "/tmp/newsapp/bestParam.txt"
    sparkSession.sparkContext.parallelize(Array(bestParam.toString(),bestMetric.toString()))
      .saveAsTextFile(bestParamPath)
    val testPreds = bestAls.transform(test)
    testPreds.createOrReplaceTempView("testPreds")
    sparkSession.sql("create table if not exists cf_cross_test (" +
      "userId int,articleId int,rating float,prediction float)")
    sparkSession.sql("insert overwrite table cf_cross_test select " +
      "userId,articleId,rating,prediction from testPreds")

  }
}
