package com.zm

import org.apache.spark.sql._
import org.apache.spark.sql.types._

//recommend for user based on every user's top 200 most similar user
object userCfRecommend {

  def main(args:Array[String]): Unit ={
    val sparkSession = SparkSession.builder()
      .appName("newsappRecommend")
      .master("local")
      .getOrCreate()
    println("start running...")

    sparkSession.sql("use newsapp")
    //userSimilarity (uid1 string,uid2 string,similarity float)
    val userSimilarity = sparkSession.sql("select * from user_similarity")
    //articlePreRec (uid string,article_id string,rating float)
    val articlePreRec = sparkSession.sql("select * from user_pre_rec")
    val userSim = userSimilarity.rdd.map(x=>(x(1),(x(0),x(1),x(2).asInstanceOf[Double])))
    val articlePre = articlePreRec.rdd.map(x=>(x(0), (x(1), x(2).asInstanceOf[Double])))
    val midPredRating = userSim.join(articlePre).map(x => {
      //key (uid,article_id)
      val key = (x._2._1._1, x._2._2._1)
      val sim = x._2._1._3
      val rating = x._2._2._2
      //stat (similarity,similarity*rating)
      val stat = (x._2._1._3, sim * rating, x._2._2._1)
      (key, stat)
    }).filter(x => x._1._1 != x._2._3)
    //generate predicted rating of user on article, get rid of those < 0
    val userPredRating = midPredRating.groupByKey().map(data => {
      val key = data._1
      val vals = data._2
      val simSum = vals.map(_._1).sum
      val ratingSum = vals.map(_._2).sum
      val finalRating = ratingSum/simSum
      (key._1,(key._2,finalRating))
    }).filter(x => x._2._2>0)
    //choose top 300 predicted ratings for every user (uid,article_id,predRating)
    val userRecArticle = userPredRating.groupByKey().flatMap(x => {
      x._2.map(y => (x._1,y._1,y._2)).toList.sortWith((left,right)=>left._3>right._3).take(300)
    }).map(data => Row(data._1.toString, data._2.toString, data._3.toFloat))
    val fields = Array(StructField("uid",StringType,nullable = true)
    ,StructField("article_id",StringType,nullable = true)
    ,StructField("predRating",FloatType, nullable = true))
    val schema = StructType(fields)
    import sparkSession.implicits._
    val userRecArticleDF = sparkSession.createDataFrame(userRecArticle, schema)
    userRecArticleDF.createOrReplaceTempView("user_rec_article")
    sparkSession.sql("create external table mid_usercf_rec_article if not exists " +
      "(uid string, article_id string, predRating float)")
    sparkSession.sql("insert overwrite table " +
      "select uid,uid,article_id,predRating from user_rec_article")

  }
}
