package com.zm

import org.apache.spark.sql._
import org.apache.spark.sql.functions.{udf,row_number,asc,desc}
import org.apache.spark.sql.expressions.Window

object labelMatchRec {

  def main(args:Array[String]): Unit ={

    val day = "'"+args(0)+"'"
    val recNums = 100
    val sparkSession = SparkSession.builder()
      .appName("newsappRecommend")
      .enableHiveSupport()
      .getOrCreate()
    sparkSession.sql("use newsapp")
    val user = sparkSession.sql("select id,label_weight from cf_user_label where length(label_weight)>0")
    val article = sparkSession.sql("select article_id,label from cf_article_label limit 10000")
    val userAct = sparkSession.sql("select id,article_id,article_time from cf_mid_user_act where day="+day)

    val userArticle = user.crossJoin(article)
    userArticle.cache()
    println(userArticle.count())
    /*val addMatch = udf(getMatchScore)
    //sparkSession.sqlContext.udf.register("getMatchScoreUDF", (text:String, labelWeight:String)=>getMatchScore(text,labelWeight))
    val userArticleMatch = userArticle
      .withColumn("matchScore",addMatch(userArticle("label"),userArticle("label_weight")))
      .where("matchScore>0")

    userArticle.unpersist()
    val userArticleUniq = userArticleMatch.join(userAct, Seq("id","article_id"),"leftouter")
      .where("article_time is null").select("id","article_id","matchScore")

    val winSpec = Window.partitionBy("id").orderBy(desc("matchScore"))
    val userArticleMatchRec = userArticleUniq
      .withColumn("rn", row_number().over(winSpec))
      .where("rn<="+recNums)
    userArticleMatchRec.createOrReplaceTempView("user_article_match_rec")
    sparkSession.sql("create table if not exists cf_user_label_match_rec "
    +"(id string,article_id string,match_score float)")
    sparkSession.sql("insert overwrite table cf_user_label_match_rec "
    +"select id,article_id,matchScore from user_article_match_rec")*/
  }

  val getMatchScore = (text:String, labelWeight:String) => {
    if (text == null || labelWeight == null){
      0f
    }

    try{
      val labelMap = labelWeight.split(",")
        .map(x=>(x.split(":")(0),x.split(":")(1).toFloat))
        .toMap
      var matchScore = 0f
      for(word <- text.split(",")){
        if(labelMap.contains(word)){
          matchScore += labelMap(word)
        }
      }
      matchScore
    }catch{
      case _:Throwable => 0f
    }
  }
}
