# Paper Sheet Break Prediction before Time of Occurence
The data contains about 18k rows collected over 15 days. T There are about 124 positive labeled sample (~0.6%).

### Data can be downloaded from
- personal_drive/data/anomaly-detection-python/processminer-rare-event-mts-data.csv

### Unbalanced data
<img src="metadata/sheetbreak_0.png">

### Training Result
<img src="metadata/sheetbreak_1.png">

### Confusion Matrix for Testing Data with 3605 normals and 49 frauds
<img src="metadata/sheetbreak_confusion_matrix.png">

### Thresholding to detect Anomaly Results
<img src="metadata/sheetbreak_result.png">

### References
https://towardsdatascience.com/extreme-rare-event-classification-using-autoencoders-in-keras-a565b386f098
