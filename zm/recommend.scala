package com.zm

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator

object recommend {

  case class Rating(userId:Int, movieId:Int, rating:Float, timestamp:Long)

  def main(args: Array[String]): Unit ={

    val day = "'"+args(0)+"'"
    val sparkSession = SparkSession.builder()
      .appName("newsappRecommend")
      .enableHiveSupport()
      .getOrCreate()

    val maxIter = 10
    val regParam = 0.1
    val userCol = "userId"
    val itemCol = "articleId"
    val ratingCol = "rating"
    val alpha = 2.0
    val rank = 200

    sparkSession.sql("use newsapp")
    val trainSamples = sparkSession.sql("select cf_uid userId,cf_article_id articleId,rating,article_time timestamp " +
      "from newsapp.cf_train_sample_filter where day="+day)
    val userPreRec = sparkSession.sql("select cf_uid userId,cf_article_id articleId,id,article_id " +
      "from cf_pre_rec_article")
    trainSamples.persist()
    userPreRec.persist()
    /*val articleNumbers = trainSamples.dropDuplicates(Array("article_id")).count()
    val userNumbers = trainSamples.dropDuplicates(Array("id")).count()*/

    val als = new ALS()
      .setMaxIter(maxIter)
      .setRegParam(regParam)
      .setAlpha(alpha)
      .setRank(rank)
      .setUserCol(userCol)
      .setItemCol(itemCol)
      .setRatingCol(ratingCol)
      .setSeed(20L)
      .setImplicitPrefs(false)
    val model = als.fit(trainSamples)
    model.setColdStartStrategy("drop")
    val userRecArticleRating = model.transform(userPreRec)
    userRecArticleRating.createOrReplaceTempView("userRecArticleRating")
    sparkSession.sql("create table if not exists cf_predict_user_article(" +
      "id string,article_id string,cf_uid int,cf_article_id int,rating_pred float)")
    sparkSession.sql("insert overwrite table cf_predict_user_article " +
      "select id,article_id,userId,articleId,prediction from userRecArticleRating")
  }
}
