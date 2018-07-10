package com.zm

/*import org.apache.log4j.Logger
import org.apache.log4j.Level*/
import org.apache.spark.sql._
import org.apache.spark.sql.types._



object sc_test {

  case class BookPair (book1:Int, book2:Int, cnt:Int, name1: String, name2: String)

  def map1(x: String): String = {
    x.replaceAll(" +",",")
  }

  val testFunc = (a:Int, b:Int) => {
    a+b
  }

  def main(args: Array[String]) {

    /*val day=args(0)
    val number = args(1)*/
    /*println("day="+day)
    println("number "+number)
    println("day type is "+day.getClass.getName)
    println("number type is "+number.getClass.getName)*/
    //Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val innerStruct =
      StructType(
        StructField("f1", IntegerType, true) ::
          StructField("f2", IntegerType, false) ::
          StructField("f3", StringType, false) :: Nil)


    val recs = Array(
      BookPair(1, 2, 3, "book1", "book2"),
      BookPair(2, 3, 1, "book2", "book3"),
      BookPair(3, 3, 2, "book1", "book3"),
      BookPair(5, 4, 5, "book1", "book4"),
      BookPair(2, 4, 7, "book2", "book4")
    )

    val recs0 = Array(
      BookPair(1, 2, 3, "book1", "book2"),
      BookPair(2, 3, 1, "book5", "book3"),
      BookPair(2, 3, 2, "book3", "book3"),
      BookPair(1, 4, 5, "book1", "book4"),
      BookPair(2, 4, 7, "book2", "book4")
    )
    val sc = SparkSession.builder().appName("mytest2").master("local").getOrCreate()
    import sc.implicits._

    val recs2 = sc.sparkContext.parallelize(recs).map(book => Row(book.book1,book.book2,book.name1))
    val recs3 = sc.sparkContext.parallelize(recs0).map(book => Row(book.book1,book.book2,book.name1))

    val fields = Array(StructField("book1",IntegerType,nullable=true), StructField("book2",IntegerType,nullable = true),
      StructField("name1",StringType,nullable = true))
    val fields2 = Array(StructField("book1",IntegerType,nullable=true), StructField("book4",IntegerType,nullable = true),
      StructField("name1",StringType,nullable = true))
    val schema = StructType(fields)
    val schema2 = StructType(fields2)
    val bookDF = sc.createDataFrame(recs2, schema)
    val bookDF2 = sc.createDataFrame(recs3, schema2)
    val bookDF3 = bookDF.join(bookDF2, Seq("name1","book1"),"leftouter").select("name1","book1","book2","book4")
    bookDF3.where("book4 is null").show()
    //bookDF3.where("book22 is null").show()
    /*val bookdf4 = bookDF3.select(bookDF("name1").alias("dfbook1"),bookDF3("name1").alias("df3book1"))
    bookdf4.show()*/
    /*bookDF.show()
    val addFunc = udf(testFunc)
    val a = bookDF.withColumn("addbook",addFunc(bookDF("book1"),bookDF("book2")))
    bookDF.show()
    a.show()*/

   /* val booksql = bookDF.groupBy($"name1").avg("book1","book2")
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
    println("recsDF type is "+recsDF.getClass.getName)*/

  }



}

