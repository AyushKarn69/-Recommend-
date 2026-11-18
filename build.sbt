name := "MovieRecommender"

version := "0.1"

// Spark 3.2.x is published for Scala 2.12 and 2.13; 2.12 matches the Spark
// binaries most clusters/Databricks runtimes ship with.
scalaVersion := "2.12.15"

val sparkVersion = "3.2.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"  % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion
)
