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
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileWriter;
import java.util.*;
import java.util.List;
/**
 *
 * Normal Data: 284315
 * Fraud Data: 492
 *
 *Training Data:
 *     Normal Data: 255884 (0 - 255883) (inclusive)
 *Testing Data:
 *     Total: 28923
 *     Normal Data: 28431 (255884 - 284314)
 *     Fraud Data: 492 (284315 - 284806)
 *
 * Data get from https://www.kaggle.com/mlg-ulb/creditcardfraud
 */
public class App
{

    static final File baseDir = new File("D:\\Users\\chiawei\\Documents\\data\\CreditCardFraud\\data\\");
    static final int numLabelClasses = 2;


    static final double threshold = 5;

    public static void main(String[] args) throws Exception
    {
        File modelSavePath = new File(baseDir, "output\\ccFraud.zip");

        File trainFeaturesDir = new File(baseDir, "train_data\\features");
        File trainLabelsDir = new File(baseDir, "train_data\\labels");

        File testFeaturesDir = new File(baseDir, "test_data\\features");
        File testLabelsDir= new File(baseDir, "test_data\\labels");

        //load training data
        int trainMinIndex = 0;
        int trainMaxIndex = 255883;

        int testMinIndex = 255884;
        int testMaxIndex = 284806;

        int miniBatchSize = 284;



        MultiLayerNetwork model = null;


        if(modelSavePath.exists())
        {
            model = ModelSerializer.restoreMultiLayerNetwork(modelSavePath);
        }
        else
        {

            Pair<SequenceRecordReader, SequenceRecordReader> rrTrain = getCreditCardDataReader(trainFeaturesDir, trainLabelsDir, trainMinIndex, trainMaxIndex);

            DataSetIterator trainIter = new SequenceRecordReaderDataSetIterator(rrTrain.getLeft(), rrTrain.getRight(), miniBatchSize, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

            /*
            DataSet Dimension
            Dimension 0: Batch Size
            Dimension 1: Feature Length Size
            Dimension 2: Time Step
            */

            int SEED = 123;
            int FEATURES_NODES = 29;
            int HIDDEN_NODES = 16;

            MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                    .seed(SEED)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam(0.005))
                    .list()
                    .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(FEATURES_NODES).nOut(HIDDEN_NODES).build())

                    .layer(1, new LSTM.Builder().activation(Activation.TANH).nIn(HIDDEN_NODES).nOut((int) (HIDDEN_NODES / 2.0)).build())

                    .layer(2, new LSTM.Builder().activation(Activation.TANH).nIn((int) (HIDDEN_NODES / 2.0)).nOut((int) (HIDDEN_NODES / 4.0)).build())

                    .layer(3, new LSTM.Builder().activation(Activation.TANH).nIn((int) (HIDDEN_NODES / 4.0)).nOut((int) (HIDDEN_NODES / 2.0)).build())

                    .layer(4, new LSTM.Builder().activation(Activation.TANH).nIn((int) (HIDDEN_NODES / 2.0)).nOut(HIDDEN_NODES).build())

                    .layer(5, new RnnOutputLayer.Builder()
                            .lossFunction(LossFunctions.LossFunction.MSE)
                            .nIn(HIDDEN_NODES).nOut(FEATURES_NODES).build())
                    .build();

            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
            uiServer.attach(statsStorage);


            model = new MultiLayerNetwork(config);
            model.init();
            model.setListeners(new StatsListener(statsStorage));//new ScoreIterationListener(10));

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

            model.save(modelSavePath, true);

            //evaluateData(model, trainIter);
        }

        Pair<SequenceRecordReader, SequenceRecordReader> rrTest = getCreditCardDataReader(testFeaturesDir, testLabelsDir, testMinIndex, testMaxIndex);
        DataSetIterator testIter = new SequenceRecordReaderDataSetIterator(rrTest.getLeft(), rrTest.getRight(), 1, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        //saveTestResult(model, testIter);
        evaluateData(model, testIter);


        System.out.println("Program end...");

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

    public static void saveTestResult(MultiLayerNetwork model, DataSetIterator iter) throws Exception
    {
        String resultFile = "CCFraudResult.csv";
        System.out.println("Saving data as " + resultFile);

        //FileCSV save Format TrueLabel, Predicted
        FileWriter csvWriter = new FileWriter(baseDir + "\\output\\" + resultFile);
        csvWriter.append("TrueLabel,Predicted,Score\n");


        iter.reset();

        while(iter.hasNext())
        {
            DataSet validData = iter.next();
            INDArray features = validData.getFeatures();
            INDArray labels = validData.getLabels();

            double score = model.score(new DataSet(features, features));

            //assume one data point per feature
            csvWriter.append(Nd4j.argMax(labels, 1).getScalar(0).toString());
            csvWriter.append(",");
            csvWriter.append("0"); //predicted label
            csvWriter.append(",");
            csvWriter.append(Double.toString(score));
            csvWriter.append("\n");

            //System.out.print(score + "   ");
        }
        csvWriter.flush();
        csvWriter.close();

        System.out.println("Save data done");

    }

    public static void evaluateData(MultiLayerNetwork model, DataSetIterator iter) throws Exception
    {

        int tp = 0; int tn = 0; int fp = 0; int fn = 0;

        List<Pair<Double, Integer>> list0 = new ArrayList<>();
        List<Pair<Double, Integer>> list1 = new ArrayList<>();

        iter.reset();

        while(iter.hasNext())
        {
            DataSet dataset = iter.next();
            INDArray features = dataset.getFeatures();
            int realLabel = Nd4j.argMax(dataset.getLabels(), 1).getInt(0);

            double score = model.score(new DataSet(features, features));
            int predictedLabel;

            if(threshold < score) predictedLabel = 1; else predictedLabel = 0;


            if( realLabel == 0) //normal //negative
            {
                list0.add(new ImmutablePair<>(score, predictedLabel));

                if(predictedLabel == 1)
                {
                    fp += 1;
                }
                else
                {
                    tn += 1;
                }
            }
            else //realLabel == 1
            {
                list1.add(new ImmutablePair<>(score, predictedLabel));

                if(predictedLabel == 1)
                {
                    tp += 1;
                }
                else
                {
                    fn += 1;
                }
            }
        }

        System.out.println("Ground Truth: TN: 28431 TP: 492");
        System.out.println("*******************************");
        System.out.print("True Negative: " + tn + " ");
        System.out.println("False Negative: " + fn);

        System.out.print("False Positive: " + fp +  " ");
        System.out.println("True Positive: " + tp);
        System.out.println("*******************************");

        Map<Integer, List<Pair<Double, Integer>>> listsByLabel = new HashMap<>(); //key =fraud, nonfraud list(score, predicted_label)

        listsByLabel.put(new Integer(0), list0);
        listsByLabel.put(new Integer(1), list1);
    }
}


//threshold setting and evaluate in python script
//show result in java (threshold + evaluation result...roc / confusion matrix)

//WORKSPACE
//CUDNN
//Manually destroying ADSI workspace
//model.setListeners(Collections.singletonList(new ScoreIterationListener(10)));
//roc curve

//confusion matrix
