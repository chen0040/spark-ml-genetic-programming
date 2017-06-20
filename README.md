# spark-ml-genetic-programming

Package provides java implementation of big-data genetic programming for Apache Spark

# Install

Add the following dependency to your POM file:

```xml
<dependency>
  <groupId>com.github.chen0040</groupId>
  <artifactId>spark-ml-genetic-programming</artifactId>
  <version>1.0.2</version>
</dependency>
```

# Features

* Linear Genetic Programming

    - Initialization
    
       + Full Register Array 
       + Fixed-length Register Array
   
    - Crossover
     
        + Linear
        + One-Point
        + One-Segment
    
    - Mutation
   
        + Micro-Mutation
        + Effective-Macro-Mutation
        + Macro-Mutation
    
    - Replacement
   
        + Tournament
        + Direct-Compete
    
    - Default-Operators
   
        + Most of the math operators
        + if-less, if-greater
        + Support operator extension
        
* Tree Genetic Programming

    - Initialization 
    
        + Full
        + Grow
        + PTC 1
        + Random Branch
        + Ramped Full
        + Ramped Grow
        + Ramped Half-Half
        
    - Crossover
    
        + Subtree Bias
        + Subtree No Bias
        
    - Mutation
    
        + Subtree
        + Subtree Kinnear
        + Hoist
        + Shrink
        
    - Evolution Strategy
    
        + (mu + lambda)
        + TinyGP


    
Future Works

* Grammar-based Genetic Programming
* Gene Expression Programming



# Usage of Linear Genetic Programming

### Create training data

The sample code below shows how to generate data from the "Mexican Hat" regression problem. We can split the data generated into training and testing data:

```java
import com.github.chen0040.gp.utils.CollectionUtils;

List<BasicObservation> data = Tutorials.mexican_hat().stream().map(s -> (BasicObservation)s).collect(Collectors.toList());
CollectionUtils.shuffle(data);
TupleTwo<List<BasicObservation>, List<BasicObservation>> split_data = CollectionUtils.split(data, 0.9);
List<BasicObservation> trainingData = split_data._1();
List<BasicObservation> testingData = split_data._2();
```
### Create and train the LGP

 
The sample code below shows how the SparkLGP can be created and trained:

```java
import com.github.chen0040.gp.lgp.LGP;
import com.github.chen0040.gp.commons.BasicObservation;
import com.github.chen0040.gp.commons.Observation;
import com.github.chen0040.gp.lgp.gp.Population;
import com.github.chen0040.gp.lgp.program.operators.*;

SparkLGP lgp = new SparkLGP();
lgp.getOperatorSet().addAll(new Plus(), new Minus(), new Divide(), new Multiply(), new Power());
lgp.getOperatorSet().addIfLessThanOperator();
lgp.addConstants(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
lgp.setRegisterCount(6); // the number of register here is the number of input dimension of the training data times 3
lgp.setPerObservationCostEvaluator((Function<Tuple2<Program, BasicObservation>, Double>) tuple2 -> {
 Program program = tuple2._1();
 BasicObservation observation = tuple2._2();
 program.execute(observation);
 return Math.pow(observation.getOutput(0) - observation.getPredictedOutput(0), 2.0);
});
lgp.setDisplayEvery(2); // display iteration result every 2 iterations


JavaSparkContext context = SparkContextFactory.createSparkContext("testing-1");
Program program = lgp.fit(context.parallelize(trainingData)); 
System.out.println(program);
```

The number of registers of a linea program is set by calling LGP.setRegisterCount(...), the number of registers is usually the a multiple of the 
input dimension of a training data instance. For example if the training data has input (x, y) which is 2 dimension, then the number of registers
may be set to 6 = 2 * 3.

The cost per observation evaluator computes the training cost of a 'program' on a particular 'observation' (which is an instance of trainingData).

The last line prints the linear program found by the LGP evolution, a sample of which is shown below:

```
instruction[1]: <If<	r[4]	c[0]	r[4]>
instruction[2]: <If<	r[3]	c[3]	r[0]>
instruction[3]: <-	r[2]	r[3]	r[2]>
instruction[4]: <*	c[7]	r[2]	r[2]>
instruction[5]: <If<	c[2]	r[3]	r[1]>
instruction[6]: </	r[1]	c[4]	r[2]>
instruction[7]: <If<	r[3]	c[7]	r[1]>
instruction[8]: <-	c[0]	r[0]	r[0]>
instruction[9]: <If<	c[7]	r[3]	r[4]>
...
```

### Test the program obtained from the LGP evolution

The best program in the LGP population obtained from the training in the above step can then be used for prediction, as shown by the sample code below:

```java
for(Observation observation : testingData) {
 program.execute(observation);
 double predicted = observation.getPredictedOutput(0);
 double actual = observation.getOutput(0);

 logger.info("predicted: {}\tactual: {}", predicted, actual);
}
```

# Usage of Tree Genetic Programming

Here we will use the "Mexican Hat" symbolic regression introduced earlier.

### Create and train the TreeGP

 
The sample code below shows how the TreeGP can be created and trained:

```java
import com.github.chen0040.gp.treegp.TreeGP;
import com.github.chen0040.gp.commons.BasicObservation;
import com.github.chen0040.gp.commons.Observation;
import com.github.chen0040.gp.treegp.gp.Population;
import com.github.chen0040.gp.treegp.program.operators.*;

SparkTreeGP tgp = new SparkTreeGP();
tgp.getOperatorSet().addAll(new Plus(), new Minus(), new Divide(), new Multiply(), new Power());
tgp.getOperatorSet().addIfLessThanOperator();
tgp.addConstants(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
tgp.setVariableCount(2); //equal to the number of input dimension of the training data
tgp.setPerObservationCostEvaluator(tuple2 -> {
 Solution program = tuple2._1();
 BasicObservation observation = tuple2._2();
 program.execute(observation);
 return Math.pow(observation.getOutput(0) - observation.getPredictedOutput(0), 2.0);
});
tgp.setDisplayEvery(2); // display iteration result every 2 iterations

JavaSparkContext context = SparkContextFactory.createSparkContext("testing-1");
Solution program = tgp.fit(context.parallelize(trainingData));  
```


The cost per observation evaluator computes the training cost of a 'program' on a particular 'observation' (which is an instance of trainingData).

The program.mathExpress() call prints the TreeGP program found by the TreeGP evolution, a sample of which is shown below:

```
Trees[0]: 1.0 - (if(1.0 < if(1.0 < 1.0, if(1.0 < v0, 1.0, 1.0), if(1.0 < (v1 * v0) + (1.0 / 1.0), 1.0 + 1.0, 1.0)), 1.0, v0 ^ 1.0))
```

### Test the program obtained from the TreeGP evolution

The best program in the TreeGP population obtained from the training in the above step can then be used for prediction, as shown by the sample code below:

```java
for(Observation observation : testingData) {
 program.execute(observation);
 double predicted = observation.getPredictedOutput(0);
 double actual = observation.getOutput(0);

 logger.info("predicted: {}\tactual: {}", predicted, actual);
}
```
