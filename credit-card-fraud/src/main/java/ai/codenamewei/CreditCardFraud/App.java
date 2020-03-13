/* *****************************************************************************
 * Copyright (c) 2020 codenamewei
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package ai.codenamewei.CreditCardFraud;

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
 * Number of dataset for total dataset:
 *      Non-Fraud Data: 284315
 *      Fraud Data    :   492
 *
 * The dataset is further partitioned into training and testing dataset.
 *
 * Number of data points for training dataset:
 *     Non-Fraud Data : 255 884 (File Index: [0.csv - 255883.csv])
 *
 * Number of data points for testing dataset:
 *     Non Fraud Data : 28431 (File Name Index: [255884.csv - 284314.csv])
 *     Fraud Data     : 492 (File Name Index: [284315.csv - 284806.csv0])
 *
 * Instructions:
 * 1. Download CreditCardFraud.zip file from this link https://drive.google.com/file/d/1ye6kjPQzt5VcQUuwLaPsxUqnAli2AoXe/view?usp=sharingv
 * 2. In App.java, set File zipFilePath to your path to CreditCardFraud.zip
 * 3. Run App.java
 *
 * @author codenamewei
 */
public class App
{
    static final int TRAIN_MIN_INDEX = 0;
    static final int TRAIN_MAX_INDEX = 255883;

    static final int TEST_MIN_INDEX = 255884;
    static final int TEST_MAX_INDEX = 284806;

    static final int SEED = 123;
    static final int FEATURE_NODES = 29;
    static final int HIDDEN_NODES = 16;

    static final int MINI_BATCH_SIZE = 284;
    static final int CLASSES = 2;

    static final double THRESHOLD = 2.5;

    public static void main(String[] args) throws Exception
    {
        //Unzip the downloaded file from the /resources/CreditCardFraud.zip into designated directory
        File zipFilePath = new File("SET\\TO\\YOUR\\PATH\\" +  "CreditCardFraud.zip");
        File baseDir = unzipFile(zipFilePath);

        //Predefined directories in the .zip data file
        File trainFeaturesDir = new File(baseDir, "\\train_feature");
        File trainLabelsDir = new File(baseDir, "\\train_label");

        File testFeaturesDir = new File(baseDir, "\\test_feature");
        File testLabelsDir= new File(baseDir, "\\test_label");

        //Load data and split into training and testing data set
        Pair<SequenceRecordReader, SequenceRecordReader> rrTrain = getCreditCardDataReader(trainFeaturesDir, trainLabelsDir, TRAIN_MIN_INDEX, TRAIN_MAX_INDEX);
        DataSetIterator trainIter = new SequenceRecordReaderDataSetIterator(rrTrain.getLeft(), rrTrain.getRight(),
                MINI_BATCH_SIZE, CLASSES, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        Pair<SequenceRecordReader, SequenceRecordReader> rrTest = getCreditCardDataReader(testFeaturesDir, testLabelsDir, TEST_MIN_INDEX, TEST_MAX_INDEX);
        DataSetIterator testIter = new SequenceRecordReaderDataSetIterator(rrTest.getLeft(), rrTest.getRight(),
                1, CLASSES, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        //Set up network configuration
        //28 -> 16 -> 8 -> 16 -> 29 (nodes)
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(FEATURE_NODES).nOut(HIDDEN_NODES).build())

                .layer(1, new LSTM.Builder().activation(Activation.TANH).nIn(HIDDEN_NODES).nOut((int) (HIDDEN_NODES / 2.0)).build())

                .layer(2, new LSTM.Builder().activation(Activation.TANH).nIn((int) (HIDDEN_NODES / 2.0)).nOut(HIDDEN_NODES).build())

                .layer(3, new RnnOutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .nIn(HIDDEN_NODES).nOut(FEATURE_NODES).build())
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

        //Save model
        File savedModelPath = new File(System.getProperty("user.home")+ "/DL4JDataDir" + "/CreditCardFraudModel.zip");
        ModelSerializer.writeModel(model, savedModelPath, false);

        System.out.println("Run evaluation on test data");
        //Evaluate the model with data of both frauds and non-frauds
        evaluateTestData(model, testIter);


        System.out.println("Program end...");
    }

    /**
     * Unzip file .zip to a designated location
     * @param zipFilePath zip file name
     * @return file path
     * @throws Exception
     */
    public static File unzipFile(File zipFilePath) throws Exception
    {
        File rootDir = new File(System.getProperty("user.home")+ "/DL4JDataDir");

        File dataDir = new File(rootDir, "/CreditCardFraudData");

        if(!dataDir.exists())
        {
            System.out.println("Unzipping data file...");

            if(!rootDir.exists()) rootDir.mkdir();

            dataDir.mkdir();

            ArchiveUtils.unzipFileTo(zipFilePath.getAbsolutePath(), dataDir.getAbsolutePath());
        }
        else
        {
            System.out.println("Data already exist. Proceed");
        }

        return dataDir.getAbsoluteFile();
    }


    /**
     * Get SequenceRecordReader for input files
     *
     * @param featureDir features home directory path
     * @param labelDir labels home directory path
     * @param minIndex minimum file index (Files index should be continuous, exp: 0.csv, 1.csv ...)
     * @param maxIndex maximum file index
     * @return Pair of SequenceRecordReader of features and labels
     * @throws Exception
     */
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

    /**
     * Evaluate model of testing data set
     * @param model trained model
     * @param iter DataSetIterator for testing data set
     */
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

            if(THRESHOLD > score) predictedLabel = 0; else predictedLabel = 1;

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
