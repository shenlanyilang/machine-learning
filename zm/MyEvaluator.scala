package com.zm

import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable, SchemaUtils}

class MyEvaluator extends BinaryClassificationEvaluator {

  override def evaluate(dataset: Dataset[_]): Double = {

    val scoreAndLabels = dataset.selectExpr("case when prediction>0 then 1 else 0 end as prediction","case when rating>0 then 1 else 0 end as rating")
    //scoreAndLabels.show(10)
    val matchResult = scoreAndLabels.selectExpr("case when prediction=rating then 1 else 0 end as matchResult")
    val  correctCnt:Double = matchResult.selectExpr("sum(matchResult) as accurateCnt")
      .rdd.first().getAs[Long]("accurateCnt").asInstanceOf[Double]
    //println("correctCnt is "+correctCnt)
    val totalNums:Double = matchResult.count()
    val accuracy = correctCnt/totalNums
    accuracy
  }

  object BinaryClassificationEvaluator extends DefaultParamsReadable[MyEvaluator] {

    override def load(path: String): MyEvaluator = super.load(path)
  }
}