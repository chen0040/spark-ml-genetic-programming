package com.github.chen0040.sparkml.gp.treegp;


import com.github.chen0040.data.utils.TupleTwo;
import com.github.chen0040.gp.commons.BasicObservation;
import com.github.chen0040.gp.commons.Observation;
import com.github.chen0040.gp.services.Tutorials;
import com.github.chen0040.gp.treegp.enums.TGPCrossoverStrategy;
import com.github.chen0040.gp.treegp.enums.TGPInitializationStrategy;
import com.github.chen0040.gp.treegp.enums.TGPMutationStrategy;
import com.github.chen0040.gp.treegp.enums.TGPPopulationReplacementStrategy;
import com.github.chen0040.gp.treegp.gp.Population;
import com.github.chen0040.gp.treegp.program.Solution;
import com.github.chen0040.gp.treegp.program.operators.*;
import com.github.chen0040.gp.utils.CollectionUtils;
import com.github.chen0040.sparkml.commons.SparkContextFactory;
import com.github.chen0040.sparkml.gp.SparkTreeGP;
import org.apache.spark.api.java.JavaSparkContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

import java.util.List;
import java.util.stream.Collectors;


/**
 * Created by xschen on 7/5/2017.
 */
public class MexicanHatUnitTest {

   private static final Logger logger = LoggerFactory.getLogger(MexicanHatUnitTest.class);



   @Test
   public void test_symbolic_regression() {

      List<BasicObservation> data = Tutorials.mexican_hat().stream().map(s -> (BasicObservation)s).collect(Collectors.toList());
      CollectionUtils.shuffle(data);
      TupleTwo<List<BasicObservation>, List<BasicObservation>> split_data = CollectionUtils.split(data, 0.9);
      List<BasicObservation> trainingData = split_data._1();
      List<BasicObservation> testingData = split_data._2();

      JavaSparkContext context = SparkContextFactory.createSparkContext("testing-1");
      SparkTreeGP tgp = createTreeGP();
      tgp.setDisplayEvery(2);

      Solution program = tgp.fit(context.parallelize(trainingData));
      logger.info("global: {}", program.mathExpression());

      test(program, testingData, false);

   }
   
   private void test(Solution program, List<BasicObservation> testingData, boolean silent) {
      for(Observation observation : testingData) {
         program.execute(observation);
         double predicted = observation.getPredictedOutput(0);
         double actual = observation.getOutput(0);

         if(!silent) {
            logger.info("predicted: {}\tactual: {}", predicted, actual);
         }
      }
   }
   
   private SparkTreeGP createTreeGP(){
      SparkTreeGP tgp = new SparkTreeGP();
      tgp.getOperatorSet().addAll(new Plus(), new Minus(), new Divide(), new Multiply(), new Power());
      tgp.getOperatorSet().addIfLessThanOperator();
      tgp.addConstants(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
      tgp.setVariableCount(2);
      tgp.setPerObservationCostEvaluator(tuple2 -> {
         Solution program = tuple2._1();
         BasicObservation observation = tuple2._2();
         program.execute(observation);
         return Math.pow(observation.getOutput(0) - observation.getPredictedOutput(0), 2.0);
      });
      tgp.setPopulationSize(1000);
      tgp.setMaxGeneration(10); // should be 1000 for full evolution
      return tgp;
   }

   @Test
   public void test_symbolic_regression_with_crossover_subtree_no_bias() {

      boolean silent = true;

      List<BasicObservation> data = Tutorials.mexican_hat().stream().map(s -> (BasicObservation)s).collect(Collectors.toList());
      CollectionUtils.shuffle(data);
      TupleTwo<List<BasicObservation>, List<BasicObservation>> split_data = CollectionUtils.split(data, 0.9);
      List<BasicObservation> trainingData = split_data._1();
      List<BasicObservation> testingData = split_data._2();

      JavaSparkContext context = SparkContextFactory.createSparkContext("testing-1");
      SparkTreeGP tgp = createTreeGP();
      tgp.setCrossoverStrategy(TGPCrossoverStrategy.CROSSVOER_SUBTREE_NO_BIAS);


      Solution program = tgp.fit(context.parallelize(trainingData));

      test(program, testingData, silent);

   }

   @Test
   public void test_symbolic_regression_with_mutation_hoist() {

      boolean silent = true;

      List<BasicObservation> data = Tutorials.mexican_hat().stream().map(s -> (BasicObservation)s).collect(Collectors.toList());
      CollectionUtils.shuffle(data);
      TupleTwo<List<BasicObservation>, List<BasicObservation>> split_data = CollectionUtils.split(data, 0.9);
      List<BasicObservation> trainingData = split_data._1();
      List<BasicObservation> testingData = split_data._2();

      JavaSparkContext context = SparkContextFactory.createSparkContext("testing-1");
      SparkTreeGP tgp = createTreeGP();
      tgp.setMutationStrategy(TGPMutationStrategy.MUTATION_HOIST);

      Solution program = tgp.fit(context.parallelize(trainingData));

      test(program, testingData, true);
   }

   @Test
   public void test_symbolic_regression_with_mutation_subtree_kinnear() {

      boolean silent = true;

      List<BasicObservation> data = Tutorials.mexican_hat().stream().map(s -> (BasicObservation)s).collect(Collectors.toList());
      CollectionUtils.shuffle(data);
      TupleTwo<List<BasicObservation>, List<BasicObservation>> split_data = CollectionUtils.split(data, 0.9);
      List<BasicObservation> trainingData = split_data._1();
      List<BasicObservation> testingData = split_data._2();

      JavaSparkContext context = SparkContextFactory.createSparkContext("testing-1");
      SparkTreeGP tgp = createTreeGP();
      tgp.setMutationStrategy(TGPMutationStrategy.MUTATION_SUBTREE_KINNEAR);

      Solution program = tgp.fit(context.parallelize(trainingData));

      test(program, testingData, silent);

   }

   @Test
   public void test_symbolic_regression_replacement_mu_plus_lambda() {

      boolean silent = true;

      List<BasicObservation> data = Tutorials.mexican_hat().stream().map(s -> (BasicObservation)s).collect(Collectors.toList());
      CollectionUtils.shuffle(data);
      TupleTwo<List<BasicObservation>, List<BasicObservation>> split_data = CollectionUtils.split(data, 0.9);
      List<BasicObservation> trainingData = split_data._1();
      List<BasicObservation> testingData = split_data._2();

      JavaSparkContext context = SparkContextFactory.createSparkContext("testing-1");
      SparkTreeGP tgp = createTreeGP();
      tgp.setReplacementStrategy(TGPPopulationReplacementStrategy.MuPlusLambda);

      Solution program = tgp.fit(context.parallelize(trainingData));

      test(program, testingData, silent);

   }




   @Test
   public void test_symbolic_regression_pop_init_ptc_1() {

      boolean silent = true;

      List<BasicObservation> data = Tutorials.mexican_hat().stream().map(s -> (BasicObservation)s).collect(Collectors.toList());
      CollectionUtils.shuffle(data);
      TupleTwo<List<BasicObservation>, List<BasicObservation>> split_data = CollectionUtils.split(data, 0.9);
      List<BasicObservation> trainingData = split_data._1();
      List<BasicObservation> testingData = split_data._2();

      JavaSparkContext context = SparkContextFactory.createSparkContext("testing-1");
      SparkTreeGP tgp = createTreeGP();
      tgp.setPopulationInitializationStrategy(TGPInitializationStrategy.INITIALIZATION_METHOD_PTC1);

      Solution program = tgp.fit(context.parallelize(trainingData));

      test(program, testingData, silent);

   }
}
