package com.codenamewei.CreditCardFraud;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;

import java.io.File;

/**
 * Data get from https://www.kaggle.com/mlg-ulb/creditcardfraud
 */
public class App
{
    public static void main(String[] args) throws Exception
    {
        String dataPath = "C:\\Users\\chiaw\\Documents\\data\\CreditCardFraud\\creditcardfraud_clean.csv";
        File inputFile = new File(dataPath);


        SequenceRecordReader reader = new CSVSequenceRecordReader();
        reader.initialize(new FileSplit(inputFile));

        int totalDataSize = 284807;
        int labelIndex = 29;
        int possibleLabels = 2;

        DataSetIterator dataIter = new SequenceRecordReaderDataSetIterator(reader, totalDataSize, possibleLabels, labelIndex);

        DataSet data1 = dataIter.next();
        int c = 0;
        /*



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

        //model.fit(trainData, trainData); //how to trainData
        */


        //WORKSPACE
        //CUDNN
    }
}


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