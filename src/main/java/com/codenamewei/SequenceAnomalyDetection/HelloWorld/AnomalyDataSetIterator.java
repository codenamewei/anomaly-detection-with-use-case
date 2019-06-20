package com.codenamewei.SequenceAnomalyDetection.HelloWorld;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

public class AnomalyDataSetIterator// implements DataSetIterator
{
    private AnomalyDataSetReader recordReader;

    public AnomalyDataSetIterator(String filePath, int batchSize)
    {
        this.recordReader = new AnomalyDataSetReader(new File(filePath));
    }
}
