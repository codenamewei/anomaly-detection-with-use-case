package com.codenamewei.SequenceAnomalyDetection.CreditCardFraud;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.util.ArchiveUtils;

import java.io.File;
/**
 *
 * This examples show Credit Card Transaction Fraud Detection.
 *
 * The patterns of the non-fraud transactions are learned and captured in order to identify
 * the outliers which can be classified as high risk for transaction frauds.
 *
 * This is accomplished in this example using a simple LSTM Autoencoder.
 * Reconstruction error should be low for stereotypical example, whereas outliers should have high reconstruction error.
 *
 * The dataset was retrieved from https://www.kaggle.com/mlg-ulb/creditcardfraud,
 * processed and dissected into smaller dataset as preprocessing step.
 *
 * The dataset is annotated with labels.
 * The features for each data point are saved in {train/test}_data_feature directory while
 * while the label is saved in {train/test}_data_label folder with the same file name.
 *
 * Training Data:
 *     Non-Fraud Data Size: 100000 (File Index: 0 - 99999) (inclusive)
 * Testing Data:
 *     Non Fraud Data Size: 8000 (Index: 0 - 7999)
 *     Fraud Data Size: 490 (Index: 8000 - 8489)
 *
 * @author ChiaWei Lim
 */
public class App
{
    static final int trainMinIndex = 0;
    static final int trainMaxIndex = 99999;

    static final int testMinIndex = 0;
    static final int testMaxIndex = 8489;

    static final int seed = 123;
    static final int featureNodes = 29;
    static final int hiddenNodes = 16;

    static final int miniBatchSize = 284;
    static final int numLabelClasses = 2;

    static final double thresholdScore= 2.5;

    public static void main(String[] args) throws Exception
    {
        //Set the train and test data paths
        File zipFilePath = new ClassPathResource("CreditCardFraud.zip").getFile();
        File baseDir = unzipFile(zipFilePath);

        File trainFeaturesDir = new File(baseDir, "\\train_data_feature");
        File trainLabelsDir = new File(baseDir, "\\train_data_label");

        File testFeaturesDir = new File(baseDir, "\\test_data_feature");
        File testLabelsDir= new File(baseDir, "\\test_data_label");

        //Load data and split into training and testing datasets
        Pair<SequenceRecordReader, SequenceRecordReader> rrTrain = getCreditCardDataReader(trainFeaturesDir, trainLabelsDir, trainMinIndex, trainMaxIndex);
        DataSetIterator trainIter = new SequenceRecordReaderDataSetIterator(rrTrain.getLeft(), rrTrain.getRight(),
                miniBatchSize, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        Pair<SequenceRecordReader, SequenceRecordReader> rrTest = getCreditCardDataReader(testFeaturesDir, testLabelsDir, testMinIndex, testMaxIndex);
        DataSetIterator testIter = new SequenceRecordReaderDataSetIterator(rrTest.getLeft(), rrTest.getRight(),
                1, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        //Set up network configuration
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(featureNodes).nOut(hiddenNodes).build())

                .layer(1, new LSTM.Builder().activation(Activation.TANH).nIn(hiddenNodes).nOut((int) (hiddenNodes / 2.0)).build())

                .layer(2, new LSTM.Builder().activation(Activation.TANH).nIn((int) (hiddenNodes / 2.0)).nOut(hiddenNodes).build())

                .layer(3, new RnnOutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .nIn(hiddenNodes).nOut(featureNodes).build())
                .build();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new StatsListener(statsStorage));

        // Train model.
        // It is worth to take note that the model is only trained with data of non-frauds to learn the general representation of non fraud data.

        System.out.println("Start Training");

        int nEpochs = 1;

        for( int i = 0; i < nEpochs; i++ )
        {
            while(trainIter.hasNext())
            {
                INDArray features = trainIter.next().getFeatures();

                model.fit(features, features);

            }
            trainIter.reset();
            System.out.println("Epoch " + i + " completed");
        }


        System.out.println("Run evaluation on test data");

        //Evaluate the model with data of both frauds and non-frauds
        evaluateTestData(model, testIter);

        System.out.println("Program end...");

    }

    public static File unzipFile(File zipFilePath) throws Exception
    {
        File rootDir = new File(System.getProperty("user.home")+ "/DL4JDataDir");

        if(!rootDir.exists()) rootDir.mkdir();

        File dataDir = new File(rootDir, "/CreditCardFraudData");

        if(!dataDir.exists())
        {
            dataDir.mkdir();
            ArchiveUtils.unzipFileTo(zipFilePath.getAbsolutePath(), dataDir.getAbsolutePath());
        }

        return dataDir.getAbsoluteFile();
    }


    public static Pair<SequenceRecordReader, SequenceRecordReader> getCreditCardDataReader(File featureDir, File labelDir, int minIndex, int maxIndex) throws Exception
    {
        int skipNumLines = 0;

        InputSplit featureFileSplit = new NumberedFileInputSplit(featureDir.getAbsolutePath() + "/%d.csv", minIndex, maxIndex);
        SequenceRecordReader features = new CSVSequenceRecordReader(skipNumLines, ",");
        features.initialize(featureFileSplit);

        InputSplit labelFileSplit = new NumberedFileInputSplit(labelDir.getAbsolutePath() + "/%d.csv", minIndex, maxIndex);
        SequenceRecordReader labels = new CSVSequenceRecordReader();
        labels.initialize(labelFileSplit);

        return new ImmutablePair<>(features, labels);
    }



    public static void evaluateTestData(MultiLayerNetwork model, DataSetIterator iter)
    {
        int tp = 0; int tn = 0; int fp = 0; int fn = 0;

        iter.reset();

        while(iter.hasNext())
        {
            DataSet dataset = iter.next();
            INDArray features = dataset.getFeatures();

            int realLabel = Nd4j.argMax(dataset.getLabels(), 1).getInt(0);

            double score = model.score(new DataSet(features, features));

            int predictedLabel;

            if(thresholdScore > score) predictedLabel = 0; else predictedLabel = 1;

            if( realLabel == 1)
            {
                if(predictedLabel == 1)
                {
                    tp += 1; // true positive
                }
                else
                {
                    fn += 1; // false negative
                }
            }
            else
            {
                if(predictedLabel == 1)
                {
                    fp += 1; // false positive
                }
                else
                {
                    tn += 1; // true negative
                }
            }

        }


        System.out.print("True Positive: " + tp + " ");
        System.out.println("False Positive: " + fp);
        System.out.print("False Negative: " + fn + " ");
        System.out.println("True Negative: " + tn);

    }
}
