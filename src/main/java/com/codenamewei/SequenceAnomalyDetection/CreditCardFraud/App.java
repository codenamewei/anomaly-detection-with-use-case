package com.codenamewei.SequenceAnomalyDetection.CreditCardFraud;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
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

/**
 * Data get from https://www.kaggle.com/mlg-ulb/creditcardfraud
 */
public class App
{

    static final File baseDir = new File("C:\\Users\\chiaw\\Documents\\data\\CreditCardFraud");
    static final int numLabelClasses = 2;

    public static void main(String[] args) throws Exception
    {
        File modelSavePath = new File(baseDir, "output\\ccFraud.zip");

        File featuresDir = new File(baseDir, "data\\features");
        File labelsDir= new File(baseDir, "data\\labels");

        //load training data
        int trainMinIndex = 0;
        int trainMaxIndex = 40000;

        int validMinIndex = 40001;
        int validMaxIndex = 45561;

        int testMinIndex = -1;
        int testMaxIndex = -1;

        int miniBatchSize = 100;


        MultiLayerNetwork model = null;


        if(modelSavePath.exists())
        {
            model = ModelSerializer.restoreMultiLayerNetwork(modelSavePath);
        }
        else
        {

            Pair<SequenceRecordReader, SequenceRecordReader> rrTrain = getCreditCardDataReader(featuresDir, labelsDir, trainMinIndex, trainMaxIndex);

            DataSetIterator dataIter = new SequenceRecordReaderDataSetIterator(rrTrain.getLeft(), rrTrain.getRight(), miniBatchSize, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

            /*
            DataSet Dimension
            Dimension 0: Batch Size
            Dimension 1: Feature Length Size
            Dimension 2: Time Step
            */

            int SEED = 123;
            int FEATURES_NODES = 29;
            int HIDDEN_NODES = 8;

            MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                    .seed(SEED)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam(0.001))
                    .list()
                    .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(FEATURES_NODES).nOut(HIDDEN_NODES).build())

                    .layer(1, new LSTM.Builder().activation(Activation.TANH).nIn(HIDDEN_NODES).nOut(HIDDEN_NODES).build())

                    .layer(2, new RnnOutputLayer.Builder()
                            .lossFunction(LossFunctions.LossFunction.MSE)
                            .nIn(HIDDEN_NODES).nOut(FEATURES_NODES).build())
                    .build();

            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
            uiServer.attach(statsStorage);


            model = new MultiLayerNetwork(config);
            model.init();
            model.setListeners(new StatsListener(statsStorage));//new ScoreIterationListener(10));

            int nEpochs = 5;
            for( int epoch = 0; epoch < nEpochs; epoch++ )
            {
                while(dataIter.hasNext())
                {
                    INDArray features = dataIter.next().getFeatures();

                    model.fit(features, features);

                }

                dataIter.reset();
                System.out.println("Epoch " + epoch + " complete");
            }

            model.save(modelSavePath, true);
        }

        evaluateData(model, featuresDir, labelsDir, validMinIndex, validMaxIndex);


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

    public static void evaluateData(MultiLayerNetwork model, File featureDir, File labelDir, int validMinIndex, int validMaxIndex) throws Exception
    {

        int validBatchSize = 1;

        Pair<SequenceRecordReader, SequenceRecordReader> rrValid = getCreditCardDataReader(featureDir, labelDir, validMinIndex, validMaxIndex);

        DataSetIterator validDataIter = new SequenceRecordReaderDataSetIterator(rrValid.getLeft(), rrValid.getRight(), validBatchSize, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);


        //FileCSV save Format TrueLabel, Predicted
        FileWriter csvWriter = new FileWriter(baseDir + "\\output\\CCFraudResult.csv");
        csvWriter.append("TrueLabel,Predicted,Score\n");


        while(validDataIter.hasNext())
        {
            DataSet validData = validDataIter.next();
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

/**
 RecordReader reader = new CSVRecordReader();
 reader.initialize(new FileSplit(inputFile));

 int totalDataSize = 284807;
 int labelIndex = 29;
 int possibleLabels = 2;

 DataSetIterator dataIter = new RecordReaderDataSetIterator(reader, totalDataSize, labelIndex, possibleLabels);

 DataSet wholeDataSet = dataIter.next();

 SplitTestAndTrain data = wholeDataSet.splitTestAndTrain(0.8);

 DataSet trainData = data.getTrain();
 DataSet testData = data.getTest();

 DataNormalization normalizer = new NormalizerStandardize();
 normalizer.fit(trainData);

 normalizer.transform(trainData);
 normalizer.transform(testData);

 int featureSize = labelIndex; //29
 int hiddenNodes = 8;
 int seed = 123;
 double learningRate = 0.01;

 MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
 .seed(seed)
 .weightInit(WeightInit.XAVIER)
 .updater(new Adam(learningRate))
 .list()
 .layer(0, new LSTM.Builder().nIn(featureSize).nOut(hiddenNodes).build())
 .layer(1, new LSTM.Builder().nIn(hiddenNodes).nOut(hiddenNodes / 2).build())
 .layer(2, new LSTM.Builder().nIn(hiddenNodes / 2).nOut(featureSize).build())
 .build();

 MultiLayerNetwork model = new MultiLayerNetwork(config);
 model.init();
 model.setListeners(new ScoreIterationListener(10));

 //model.fit(trainData, trainData); //how to trainData*/