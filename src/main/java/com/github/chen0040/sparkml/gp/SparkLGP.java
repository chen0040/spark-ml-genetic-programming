package com.github.chen0040.sparkml.gp;


import com.github.chen0040.gp.commons.BasicObservation;
import com.github.chen0040.gp.lgp.LGP;
import com.github.chen0040.gp.lgp.gp.Population;
import com.github.chen0040.gp.lgp.program.Program;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import scala.Tuple2;


/**
 * Created by xschen on 6/6/2017.
 */
public class SparkLGP extends LGP {

   private JavaRDD<BasicObservation> observationRdd;

   private Function<Tuple2<Program, BasicObservation>, Double> perObservationCostEvaluator;

   public void setPerObservationCostEvaluator(Function<Tuple2<Program, BasicObservation>, Double> perObservationCostEvaluator) {
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
   public double evaluateCost(Program program) {
      program.markStructuralIntrons(this);
      program = program.makeEffectiveCopy();

      JavaSparkContext context = JavaSparkContext.fromSparkContext(observationRdd.context());
      Broadcast<Program> programBroadcast = context.broadcast(program);
      Tuple2<Double, Integer> result = observationRdd.map(observation -> {
         Program p = programBroadcast.getValue();
         return new Tuple2<>(p, observation);
      }).map(perObservationCostEvaluator)
              .map(a -> new Tuple2<>(a, 1))
              .reduce((a, b) -> new Tuple2<>(a._1() + b._1(), a._2() + b._2()));
      double cost = result._1() / result._2();
      programBroadcast.destroy();
      return cost;
   }

   public Program fit(JavaRDD<BasicObservation> trainingData) {
      setObservationRdd(trainingData);

      long startTime = System.currentTimeMillis();
      Population pop = this.newPopulation();
      pop.initialize();
      while (!pop.isTerminated())
      {
         pop.evolve();
         if(getDisplayEvery() > 0 && pop.getCurrentGeneration() % getDisplayEvery() == 0) {
            long seconds = (System.currentTimeMillis() - startTime) / 1000;
            System.out.println("Generation: " + pop.getCurrentGeneration() + " (Pop: " + pop.size() + "), elapsed: " + seconds + " seconds");
            System.out.println("Global Cost: " + pop.getGlobalBestProgram().getCost() + "\tCurrent Cost: " + pop.getCostInCurrentGeneration());
         }
      }
      return pop.getGlobalBestProgram();
   }
}
