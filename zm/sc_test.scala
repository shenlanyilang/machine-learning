package com.zm

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.shared._


object sc_test {

  case class BookPair (book1:Int, book2:Int, cnt:Int, name1: String, name2: String)

  def map1(x: String): String = {
    x.replaceAll(" +",",")
  }

  def main(args: Array[String]) {

    /*val day=args(0)
    val number = args(1)*/
    /*println("day="+day)
    println("number "+number)
    println("day type is "+day.getClass.getName)
    println("number type is "+number.getClass.getName)*/
    val innerStruct =
      StructType(
        StructField("f1", IntegerType, true) ::
          StructField("f2", IntegerType, false) ::
          StructField("f3", StringType, false) :: Nil)


    val recs = Array(
      BookPair(1, 2, 3, "book1", "book2"),
      BookPair(2, 3, 1, "book2", "book3"),
      BookPair(1, 3, 2, "book1", "book3"),
      BookPair(1, 4, 5, "book1", "book4"),
      BookPair(2, 4, 7, "book2", "book4")
    )

    val sc = SparkSession.builder().appName("mytest2").getOrCreate()
    import sc.implicits._

    val recs2 = sc.sparkContext.parallelize(recs).map(book => Row(book.book1,book.book2,book.name1))
    println("rec2 first row type is "+recs2.getClass.getName)
    //val fields = "book1,book2,name1".split(",").map(name => StructField(name,StringType,nullable = true))
    val fields = Array(StructField("book1",IntegerType,nullable=true), StructField("book2",IntegerType,nullable = true),
      StructField("name1",StringType,nullable = true))
    val schema = StructType(fields)
    val bookDF = sc.createDataFrame(recs2, schema)
    val book0 = bookDF.selectExpr("case when book1>1 then 1 else 0 end as book0")
    book0.printSchema()
    println("book0 data is ")
    book0.show()
    println("bookDF data is:")
    bookDF.show()
    val bookRow = bookDF.rdd.first()
    println("bookRow first row is "+bookRow)
    println("bookRow first type is "+bookRow.getClass.getName)

    val booksql = bookDF.groupBy($"name1").avg("book1","book2")
    booksql.show()

    val recsRdd = sc.sparkContext.parallelize(recs)
    println("recsRdd type is "+recsRdd.getClass.getName)
    val recsDF = recsRdd.toDF()
    recsDF.printSchema()
    recsDF.select("book1").show()
    recsDF.createOrReplaceTempView("books")
    val book1 = recsDF.map(x => "book1"+x.getAs("book1"))
    println("book1 type is "+book1.getClass.getName)
    book1.show(20)
    val sqlDF = sc.sql("select book1,book2,substr(name1,1,2) name1,substr(name2,1,3) name2 from books")
    sqlDF.show()
    println(recsDF.first())
    println("recsDF_rdd type is "+recsDF.rdd.first().getClass.getName)
    println("recsDF_rdd first row is "+recsDF.rdd.first())


    println("recsDF type is "+recsDF.getClass.getName)

    /*val movieFile = "D:\\spark_local\\ml-100k\\u.data" // Should be some file on your system
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local")//.setJars(List("/home/zhuan/IdeaProjects/hello/out/artifacts/hello_jar/hello.jar"))
    val sc = new SparkContext(conf)
    val logData = sc.textFile(movieFile, 2).cache()
    println("logData type is "+logData.getClass)
    println("data lines number is "+logData.count())
    //println(logData.collect())
    val numAs = logData.filter(line => line.contains("a")).count()
    val numBs = logData.filter(line => line.contains("b")).count()
    println("Lines with a: %s, Lines with b: %s".format(numAs, numBs))*/

    ////    val data = Array(1, 2, 3, 4, 5)
    ////    val distData = sc.parallelize(data)
    ////    println(distData.take(1))
    ////    println(distData)
    //    val lines = sc.textFile("/home/zhuan/Soft/spark-2.0.0-bin-hadoop2.7/README.md")
    //    val lineLengths = lines.map(s => s.length)
    //    val totalLength = lineLengths.reduce((a, b) => a + b)
    //    println(totalLength)



  }

}

