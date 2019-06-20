package com.codenamewei.SequenceAnomalyDetection.HelloWorld;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**Detection anomaly data with sequence data sensors
 *`
 * The model is build only with normal data to model the distribution
 *
 * The normal data should have low reconfiguration error,
 * whereas those anomaly data which the autoencoder has not encountered  have high reconstruction error
 */
public class App
{
    public static Logger logger = LoggerFactory.getLogger(App.class);

    private static int trainBatchSize = 64;
    private static int testBatchSize = 1;

    private static int numEpochs = 50;

    public static void main( String[] args ) throws Exception
    {
        String filePath = new ClassPathResource("/anomalysequencedata").getFile().getPath();

        //Read in Data
        String trainFilePath = filePath + File.separator + "sensordata_train.csv";
        String testFilePath = filePath + File.separator + "sensordata_test.csv";

        //DataSetIterator trainIterator = new AnomalyDataSetIterator(trainFilePath, trainBatchSize);
        //DataSetIterator testIterator = new AnomalyDataSetIterator(testFilePath, testBatchSize);

        logger.info("-----------Program End--------------");


    }
}
