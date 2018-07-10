package com.zm

import org.apache.spark.sql._
import org.apache.spark.sql.types._

//calculate users' similarity and generate every user's top 200 most similar user
object userSimGen {
  def main(args: Array[String]): Unit ={

    val sparkSession = SparkSession.builder()
      .appName("newsappRecommend")
      .master("master")
      .enableHiveSupport()
      .getOrCreate()

    sparkSession.sql("use newsapp")
    val trainSamples = sparkSession.sql("select cast(id as int) userId,cf_article_id articleId,rating,article_time timestamp " +
      "from newsapp.cf_train_sample where id like 'user_%' and day=2018-06-26")
    val ratingOrig = trainSamples.selectExpr("userId","articleId","rating")
    val ratings = trainSamples.selectExpr("userId","articleId","rating").rdd
      .map(rating => (rating.getAs[Int]("articleId")
        ,rating.getAs[Int]("userId")
        ,rating.getAs[Float]("rating")))
    val userArticleNum = ratings.groupBy(x => x._2).map(x => (x._1,x._2.size)).keyBy(x=>x._1)
    //val ratingTransform1 = ratings.keyBy(x=>x._1)
    val ratingTransform1 = ratings.groupBy(_._2).flatMap(x=>{
      val size = x._2.size
      val mapresult = x._2.map(y=>(y._1,y._2,y._3,size))
      mapresult
    }).keyBy(_._1)
    val ratingTransform2 = ratingTransform1.join(ratingTransform1).filter(x=> x._2._1._2>x._2._2._2)
    val ratingTransform3 = ratingTransform2.map(x=>{
      val key = (x._2._1._1,x._2._2._1)
      val rating1 = x._2._1._3
      val rating2 = x._2._2._3
      val rating1Square = rating1*rating1
      val rating2Square = rating2*rating2
      val ratingMulti = rating1*rating2
      val size1 = x._2._1._4
      val size2 = x._2._2._4
      val stat = (rating1,rating2,rating1Square,rating2Square,ratingMulti,size1,size2)
      (key,stat)
    })
    val ratingTransform4 = ratingTransform3.groupByKey()
      .map(data => {
        val key = data._1
        val vals = data._2
        val size = vals.size
        val dotProduct = vals.map(x => x._5).sum
        val rating1Sum = vals.map(x => x._1).sum
        val rating2Sum = vals.map(x => x._2).sum
        val rating1Sq = vals.map(x => x._3).sum
        val rating2Sq = vals.map(x => x._4).sum
        val numRaters1 = vals.map(x => x._6).max
        val numRaters2 = vals.map(x => x._7).max
        val stat = (dotProduct, rating1Sum, rating2Sum, rating1Sq, rating2Sq
        ,numRaters1, numRaters2, size)
        (key,stat)
      })
    val inverseRatingTransformed4 = ratingTransform4.map(stats => {
      val key = (stats._1._2,stats._1._1)
      val vals = stats._2
      val size = vals._8
      val dotProduct = vals._1
      val rating1Sum = vals._3
      val rating2Sum = vals._2
      val rating1Sq = vals._5
      val rating2Sq = vals._4
      val numRaters1 = vals._7
      val numRaters2 = vals._6
      val stat = (dotProduct, rating1Sum, rating2Sum, rating1Sq, rating2Sq
      ,numRaters1, numRaters2, size)
      (key,stat)
    })
    val combineRatings = ratingTransform4 ++ inverseRatingTransformed4
    val similarities = combineRatings.map(fields => {
      val key = fields._1
      val (dotProduct, rating1Sum, rating2Sum, rating1Sq, rating2Sq, numRaters1, numRaters2, size) = fields._2
      /*val cosSim = modiCosineSimilarity(dotProduct*size, numRaters1*math.sqrt(rating1Sq), math.sqrt(rating2Sq)*math.log10(10+numRaters2))
      (key._1,(key._2,cosSim))
    })*/
      val cosSim = modiCosineSimilarity(dotProduct, math.sqrt(rating1Sq), math.sqrt(rating2Sq))
      (key._1,(key._2,cosSim))
      })

    //choose top 200 most similar user
    val similarityTop = similarities.groupByKey().flatMap(sim => {
     sim._2.map(x => (sim._1,(x._1,x._2))).toList.sortWith((left,right)=>left._2._2>right._2._2).take(200)
    })

    val similarity2Row = similarityTop.map(sim => Row(sim._1.toString, sim._2._1.toString, sim._2._2.toFloat))
    val fields = Array(StructField("user1",StringType, nullable=true),StructField("user2",StringType,nullable = true)
    ,StructField("similarity",FloatType, nullable=true))
    val schema = StructType(fields)
    val similirity2DF = sparkSession.createDataFrame(similarity2Row,schema)
    //write to hive table
    similirity2DF.createOrReplaceTempView("tmp_user_similarity")
    sparkSession.sql("create external table if not exists user_similarity " +
      "(uid1 string,uid2,string,similarity float)")
    sparkSession.sql("insert overwrite table user_similarity " +
      "select user1,user2,similarity from tmp_user_similarity")

  }

  def modiCosineSimilarity(dotProduct:Double,rating1Norm:Double, rating2Norm:Double)= {
    if(rating1Norm == 0 || rating2Norm == 0){
      0.0
    }else {
      dotProduct / (rating1Norm * rating2Norm)
    }
  }

}
