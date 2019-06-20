package com.codenamewei.SequenceAnomalyDetection.HelloWorld;

import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import java.io.File;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class AnomalyDataSetReader
{
    private int skipNumLines;
    private int skipNumColumns;
    private int longestTimeSequence;
    private int shortest;

    private Path filePath;
    private Queue<String> currentLines;

    private Iterator<List<Writable>> iter;
    private int totalExamples;

    public AnomalyDataSetReader(File file)
    {
        this.skipNumLines = 1;
        this.skipNumColumns = 2;

        this.longestTimeSequence = 0;
        this.shortest = 1;

        this.filePath = file.toPath();
        this.currentLines = new LinkedList<String>();

        init();
    }

    public void init()
    {
        List<List<Writable>> dataLines = new ArrayList<>();

        try
        {
            List<String> lines = Files.readAllLines(filePath, Charset.forName("UTF-8"));

            for (int i = skipNumLines; i < lines.size(); i ++)
            {
                String tempStr = lines.get(i).replaceAll("\"", "");
                currentLines.offer(tempStr);

                int templength = tempStr.split(",").length - skipNumColumns;

                longestTimeSequence = longestTimeSequence < templength? templength:longestTimeSequence;
                List<Writable> dataLine = new ArrayList<>();

                String[] wary = tempStr.split(",");
                for (int j = skipNumColumns; j < wary.length; j++ ) {
                    dataLine.add(new Text(wary[j]));
                }
                dataLines.add(dataLine);
            }
        }
        catch(Exception e)
        {
            throw new RuntimeException("loading data failed");
        }

        iter = dataLines.iterator();
        totalExamples = dataLines.size();
    }
}
