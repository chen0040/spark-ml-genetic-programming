package com.github.chen0040.sparkml.gp.lgp;


import com.github.chen0040.data.utils.TupleTwo;
import com.github.chen0040.gp.commons.BasicObservation;
import com.github.chen0040.gp.commons.Observation;
import com.github.chen0040.gp.lgp.LGP;
import com.github.chen0040.gp.lgp.enums.LGPCrossoverStrategy;
import com.github.chen0040.gp.lgp.gp.Population;
import com.github.chen0040.gp.lgp.program.Program;
import com.github.chen0040.gp.lgp.program.operators.*;
import com.github.chen0040.gp.utils.CollectionUtils;
import com.github.chen0040.sparkml.commons.SparkContextFactory;
import com.github.chen0040.sparkml.gp.SparkLGP;
import com.github.chen0040.sparkml.gp.utils.FileUtil;
import org.apache.spark.api.java.JavaSparkContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.AssertionsForInterfaceTypes.assertThat;


/**
 * Created by xschen on 10/5/2017.
 */
public class SpiralClassificationUnitTest {

   private static final Logger logger = LoggerFactory.getLogger(SpiralClassificationUnitTest.class);

   @Test
   public void test_symbolic_classification() throws IOException {
      List<BasicObservation> data = spiral();
      CollectionUtils.shuffle(data);
      TupleTwo<List<BasicObservation>, List<BasicObservation>> split_data = CollectionUtils.split(data, 0.8);
      List<BasicObservation> trainingData = split_data._1();
      List<BasicObservation> testingData = split_data._2();

      JavaSparkContext context = SparkContextFactory.createSparkContext("testing-1");
      SparkLGP lgp = createLGP();
      lgp.setObservationRdd(context.parallelize(trainingData));

      Population pop = train(lgp);

      Program program = pop.getGlobalBestProgram();
      logger.info("global: {}", program);

      test(program, testingData);

      testMakeCopy(program);
   }

   private void testMakeCopy(Program program) {
      Program copy = program.makeCopy();
      assertThat(copy).isEqualTo(program);
      assertThat(copy.hashCode()).isEqualTo(program.hashCode());
   }

   private Population train(LGP lgp) {
      long startTime = System.currentTimeMillis();
      Population pop = lgp.newPopulation();
      pop.initialize();
      while (!pop.isTerminated())
      {
         pop.evolve();
         logger.info("Spiral Symbolic Classification Generation: {}, elapsed: {} seconds", pop.getCurrentGeneration(), (System.currentTimeMillis() - startTime) / 1000);
         logger.info("Global Cost: {}\tCurrent Cost: {}", pop.getGlobalBestProgram().getCost(), pop.getCostInCurrentGeneration());
      }

      return pop;
   }

   private void test(Program program, List<BasicObservation> testingData) {
      for(Observation observation : testingData) {
         program.execute(observation);
         int predicted = observation.getPredictedOutput(0) > 0.5 ? -1 : 1;
         int actual = observation.getOutput(0) > 0.5 ? -1 : 1;

         logger.info("predicted: {}\tactual: {}", predicted, actual);
      }
   }

   private SparkLGP createLGP(){
      SparkLGP lgp = new SparkLGP();
      lgp.getOperatorSet().addAll(new Plus(), new Minus(), new Divide(), new Multiply(), new Sine(), new Cosine());
      lgp.getOperatorSet().addIfLessThanOperator();
      lgp.addConstants(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
      lgp.setRegisterCount(6);
      lgp.setPerObservationCostEvaluator(tuple2 -> {
         Program program = tuple2._1();
         BasicObservation observation = tuple2._2();
         program.execute(observation);

         int actual = observation.getOutput(0) > 0.5 ? -1 : 1;
         int predicted = observation.getPredictedOutput(0) > 0.5 ? -1 : 1;
         return actual != predicted ? 1.0 : 0;
      });
      lgp.setMaxGeneration(30); // should be 1000 for full evolution
      lgp.setCrossoverStrategy(LGPCrossoverStrategy.OneSegment);
      lgp.setMaxProgramLength(200);
      lgp.setMinProgramLength(1);

      return lgp;
   }

   private List<BasicObservation> spiral() throws IOException {
      List<BasicObservation> result = new ArrayList<>();

      InputStream inputStream = FileUtil.getResource("spiral-dataset.txt");

      try(BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))){
         String line;
         boolean firstLine = true;
         while((line = reader.readLine()) != null){
            if(firstLine){
               firstLine = false;
               continue;
            }

            String[] parts = line.split("\t");
            double x = Double.parseDouble(parts[0]);
            double y = Double.parseDouble(parts[1]);
            int label = Integer.parseInt(parts[2]);

            BasicObservation observation = new BasicObservation(2, 2);

            observation.setInput(0, x);
            observation.setInput(1, y);
            observation.setOutput(0, label == -1 ? 1 : 0);
            observation.setOutput(1, label != -1 ? 1 : 0);

            result.add(observation);

         }
      }catch(IOException ioe){
         logger.error("Failed to read spiral-dataset.txt", ioe);
      }



      return result;
   }
}
