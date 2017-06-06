package com.github.chen0040.sparkml.gp;


import com.github.chen0040.gp.commons.BasicObservation;
import com.github.chen0040.gp.lgp.LGP;
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

   private Function<Tuple2<Program, BasicObservation>, Double> costEvaluator;

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
      double cost = observationRdd.map(observation -> {
         Program p = programBroadcast.getValue();
         return new Tuple2<>(p, observation);
      }).map(costEvaluator).reduce((a, b) -> a + b);
      programBroadcast.destroy();
      return cost;
   }
}
