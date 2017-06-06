package com.github.chen0040.sparkml.gp;


import com.github.chen0040.gp.commons.BasicObservation;
import com.github.chen0040.gp.treegp.TreeGP;
import com.github.chen0040.gp.treegp.program.Program;
import com.github.chen0040.gp.treegp.program.Solution;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import scala.Tuple2;


/**
 * Created by xschen on 6/6/2017.
 */
public class SparkTreeGP extends TreeGP {

   private JavaRDD<BasicObservation> observationRdd;

   private Function<Tuple2<Solution, BasicObservation>, Double> perObservationCostEvaluator;

   public void setPerObservationCostEvaluator(Function<Tuple2<Solution, BasicObservation>, Double> perObservationCostEvaluator) {
      this.perObservationCostEvaluator = perObservationCostEvaluator;
   }

   public void setObservationRdd(JavaRDD<BasicObservation> observationRdd) {
      setObservationRdd(observationRdd, -1);
   }

   public void setObservationRdd(JavaRDD<BasicObservation> observationRdd, int partitionCount) {
      if(partitionCount == -1) {
         this.observationRdd = observationRdd.cache();
      } else {
         this.observationRdd = observationRdd.coalesce(partitionCount).cache();
      }
   }

   @Override
   public int getTreeCountPerSolution(){
      return observationRdd.first().outputCount();
   }

   @Override
   public double evaluateCost(Solution solution) {
      JavaSparkContext context = JavaSparkContext.fromSparkContext(observationRdd.context());
      Broadcast<Solution> solutionBroadcast = context.broadcast(solution);
      double cost = observationRdd.map(observation -> {
         Solution p = solutionBroadcast.getValue();
         return new Tuple2<>(p, observation);
      }).map(perObservationCostEvaluator).reduce((a, b) -> a + b);
      solutionBroadcast.destroy();
      return cost;
   }

}
