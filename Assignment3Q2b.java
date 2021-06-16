package Assignment3;

// Assignment 3 :   Tools & Techniques for Large Scale Data Analytics
// Name         :   Rahul Kumar
// Student ID   :   20230113

// Q3.  Compute the Area Under the ROC Curve using the complete testing data.

// Importing the "java.util" package for taking input from Scanner class
// Importing the spark packages
// Importing mllib packages

import java.util.Arrays;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;


public final class Assignment3Q2b { 
    public static void main(String[] args) throws Exception {
        // setting up the Spark configuration and Spark context
        System.setProperty("hadoop.home.dir", "C:/winutils");
        SparkConf sparkConf =   new SparkConf()
                                    .setAppName("WordCount")
                                    .setMaster("local[4]")
                                    .set("spark.executor.memory", "1g");

        JavaSparkContext ctx = new JavaSparkContext(sparkConf);
            
        System.out.println("\n\n \t\t\t\t Assignment3Q2b\n\n\n ");
            
        // path where the txt file is stored
        String path = "C:/Users/rahul/Desktop/Codes/NUIG Assignment Codes/5. LDA/Assignment 3/imdb_labelled.txt";
        
        // Reading the txt file in JavaRDD
        JavaRDD<String> text = ctx.textFile(path, 1);
        
        // Create a HashingTF instance to map txt file string(movie review) to vectors of 100 features
        final HashingTF tf = new HashingTF(100);
        
        // each string is split(by tab) into vectors and labels for the labeledpoints
        // vector are transformed into numerical feature vector using tf(HashingTF)
        JavaRDD<LabeledPoint> labeledtextRDD = text.map(line -> {
                String[] values = line.split("\\t");
                return new LabeledPoint(Double.parseDouble(values[1]), tf.transform(Arrays.asList(values[0].trim().split(" "))));
        });
            
        // Split labeledtextRDD into two... [60% training data, 40% testing data].
        JavaRDD<LabeledPoint> training = labeledtextRDD.sample(false, 0.6);
        training.cache(); // cache training data as needed to build the model
        JavaRDD<LabeledPoint> test = labeledtextRDD.sample(false, 0.4);
            
        // Run training algorithm to build the model.
        int numIterations = 1000;
        SVMModel model = SVMWithSGD.train(training.rdd(), numIterations);

        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test.map(p -> 
        new Tuple2<>(model.predict(p.features()), p.label()));

        // Get evaluation metrics.
        BinaryClassificationMetrics metrics =
        new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
        double auROC = metrics.areaUnderROC();
                
        System.out.println("\n\n\n\nArea under ROC = " + auROC + "\n\n\n");
        
        ctx.stop();
        ctx.close();
    }
}           