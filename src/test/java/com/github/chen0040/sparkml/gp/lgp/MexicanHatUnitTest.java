package com.github.chen0040.sparkml.gp.lgp;


import com.github.chen0040.data.utils.TupleTwo;
import com.github.chen0040.gp.commons.BasicObservation;
import com.github.chen0040.gp.commons.Observation;
import com.github.chen0040.gp.lgp.LGP;
import com.github.chen0040.gp.lgp.enums.LGPCrossoverStrategy;
import com.github.chen0040.gp.lgp.enums.LGPInitializationStrategy;
import com.github.chen0040.gp.lgp.enums.LGPReplacementStrategy;
import com.github.chen0040.gp.lgp.gp.Population;
import com.github.chen0040.gp.lgp.program.Program;
import com.github.chen0040.gp.lgp.program.operators.*;
import com.github.chen0040.gp.utils.CollectionUtils;
import com.github.chen0040.sparkml.commons.SparkContextFactory;
import com.github.chen0040.sparkml.gp.SparkLGP;
import com.github.chen0040.sparkml.gp.utils.ProblemCatalogue;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;
import scala.Tuple2;

import java.util.List;


/**
 * Created by xschen on 7/5/2017.
 */
public class MexicanHatUnitTest {

   private static final Logger logger = LoggerFactory.getLogger(MexicanHatUnitTest.class);


   @Test
   public void test_symbolic_regression() {

      boolean silent = false;
      
      List<BasicObservation> data = ProblemCatalogue.mexican_hat();
      CollectionUtils.shuffle(data);
      TupleTwo<List<BasicObservation>, List<BasicObservation>> split_data = CollectionUtils.split(data, 0.9);
      List<BasicObservation> trainingData = split_data._1();
      List<BasicObservation> testingData = split_data._2();

      JavaSparkContext context = SparkContextFactory.createSparkContext("testing-1");
      SparkLGP lgp = createLGP();
      lgp.setObservationRdd(context.parallelize(trainingData));

      Population pop = train(lgp, silent);

      Program program = pop.getGlobalBestProgram();
      logger.info("global: {}", program);

      test(program, testingData, silent);

   }
   
   private void test(Program program, List<BasicObservation> testingData, boolean silent) {
      for(Observation observation : testingData) {
         program.execute(observation);
         double predicted = observation.getPredictedOutput(0);
         double actual = observation.getOutput(0);

         if(!silent) {
            logger.info("predicted: {}\tactual: {}", predicted, actual);
         }
      }
   }
   
   private SparkLGP createLGP(){
      SparkLGP lgp = new SparkLGP();
      lgp.getOperatorSet().addAll(new Plus(), new Minus(), new Divide(), new Multiply(), new Power());
      lgp.getOperatorSet().addIfLessThanOperator();
      lgp.addConstants(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
      lgp.setRegisterCount(6);

      lgp.setPerObservationCostEvaluator((Function<Tuple2<Program, BasicObservation>, Double>) tuple2 -> {
         Program program = tuple2._1();
         BasicObservation observation = tuple2._2();
         program.execute(observation);
         return Math.pow(observation.getOutput(0) - observation.getPredictedOutput(0), 2.0);
      });
      lgp.setMaxGeneration(30); // should be 1000 for full evolution
      return lgp;
   }

   private Population train(LGP lgp, boolean silent) {
      long startTime = System.currentTimeMillis();
      Population pop = lgp.newPopulation();
      pop.initialize();
      while (!pop.isTerminated())
      {
         pop.evolve();
         if(!silent) {
            logger.info("Mexican Hat Symbolic Regression Generation: {} (Pop: {}), elapsed: {} seconds", pop.getCurrentGeneration(),
                    pop.size(),
                    (System.currentTimeMillis() - startTime) / 1000);
            logger.info("Global Cost: {}\tCurrent Cost: {}", pop.getGlobalBestProgram().getCost(), pop.getCostInCurrentGeneration());
         }
      }

      return pop;
   }

   @Test
   public void test_symbolic_regression_with_crossover_onePoint() {

      boolean silent = true;

      List<BasicObservation> data = ProblemCatalogue.mexican_hat();
      CollectionUtils.shuffle(data);
      TupleTwo<List<BasicObservation>, List<BasicObservation>> split_data = CollectionUtils.split(data, 0.9);
      List<BasicObservation> trainingData = split_data._1();
      List<BasicObservation> testingData = split_data._2();

      JavaSparkContext context = SparkContextFactory.createSparkContext("testing-1");
      SparkLGP lgp = createLGP();
      lgp.setObservationRdd(context.parallelize(trainingData));
      lgp.setCrossoverStrategy(LGPCrossoverStrategy.OnePoint);

      Population pop = train(lgp, silent);

      Program program = pop.getGlobalBestProgram();

      test(program, testingData, silent);

   }

   @Test
   public void test_symbolic_regression_with_crossover_oneSegment() {

      boolean silent = true;

      List<BasicObservation> data = ProblemCatalogue.mexican_hat();
      CollectionUtils.shuffle(data);
      TupleTwo<List<BasicObservation>, List<BasicObservation>> split_data = CollectionUtils.split(data, 0.9);
      List<BasicObservation> trainingData = split_data._1();
      List<BasicObservation> testingData = split_data._2();

      JavaSparkContext context = SparkContextFactory.createSparkContext("testing-1");
      SparkLGP lgp = createLGP();
      lgp.setObservationRdd(context.parallelize(trainingData));
      lgp.setCrossoverStrategy(LGPCrossoverStrategy.OneSegment);

      Population pop = train(lgp, silent);

      Program program = pop.getGlobalBestProgram();

      test(program, testingData, silent);

   }

   @Test
   public void test_symbolic_regression_replacement_direct_compete() {

      boolean silent = true;

      List<BasicObservation> data = ProblemCatalogue.mexican_hat();
      CollectionUtils.shuffle(data);
      TupleTwo<List<BasicObservation>, List<BasicObservation>> split_data = CollectionUtils.split(data, 0.9);
      List<BasicObservation> trainingData = split_data._1();
      List<BasicObservation> testingData = split_data._2();

      JavaSparkContext context = SparkContextFactory.createSparkContext("testing-1");
      SparkLGP lgp = createLGP();
      lgp.setObservationRdd(context.parallelize(trainingData));
      lgp.setReplacementStrategy(LGPReplacementStrategy.DirectCompetition);

      Population pop = train(lgp, silent);

      Program program = pop.getGlobalBestProgram();

      test(program, testingData, silent);

   }


   @Test
   public void test_symbolic_regression_effective_mutation() {

      boolean silent = true;

      List<BasicObservation> data = ProblemCatalogue.mexican_hat();
      CollectionUtils.shuffle(data);
      TupleTwo<List<BasicObservation>, List<BasicObservation>> split_data = CollectionUtils.split(data, 0.9);
      List<BasicObservation> trainingData = split_data._1();
      List<BasicObservation> testingData = split_data._2();

      JavaSparkContext context = SparkContextFactory.createSparkContext("testing-1");
      SparkLGP lgp = createLGP();
      lgp.setObservationRdd(context.parallelize(trainingData));
      lgp.setEffectiveMutation(true);

      Population pop = train(lgp, silent);

      Program program = pop.getGlobalBestProgram();

      test(program, testingData, silent);

   }

   @Test
   public void test_symbolic_regression_pop_init_const_length() {

      boolean silent = true;

      List<BasicObservation> data = ProblemCatalogue.mexican_hat();
      CollectionUtils.shuffle(data);
      TupleTwo<List<BasicObservation>, List<BasicObservation>> split_data = CollectionUtils.split(data, 0.9);
      List<BasicObservation> trainingData = split_data._1();
      List<BasicObservation> testingData = split_data._2();

      JavaSparkContext context = SparkContextFactory.createSparkContext("testing-1");
      SparkLGP lgp = createLGP();
      lgp.setObservationRdd(context.parallelize(trainingData));
      lgp.setProgramInitializationStrategy(LGPInitializationStrategy.ConstantLength);

      Population pop = train(lgp, silent);

      Program program = pop.getGlobalBestProgram();

      test(program, testingData, silent);

   }
}
