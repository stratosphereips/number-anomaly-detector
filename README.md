# Number Anomaly Detector

- Author: Sebastian Garcia (eldraco@gmail.com, @eldracote)

An anomaly detector for numbers. 
You can give a file with one number per line and it will print the anomalies.
You can also give numbers in the STDIN and it will print the anomalies every X amount of numbers.

## Dependencies

Please install the following dependencies:
- pyod: PyOD is a comprehensive and scalable Python toolkit for detecting outlying objects in multivariate data. 

```
pip install pyod
```

## Usage

### From a file
```
$ ./number_anomaly_detector.py -f test-numbers.txt
Simple Number Anomaly Detector. Version: 0.1
Author: Sebastian Garcia (eldraco@gmail.com)

Top anomalies
values    score
35345 3.129754
24562 1.766415
2 1.338806
```

### From STDIN
```
$ cat test-numbers.txt | ./number_anomaly_detector.py
Simple Number Anomaly Detector. Version: 0.1
Author: Sebastian Garcia (eldraco@gmail.com)

Top anomalies
values    score
 35345 1.999886

Top anomalies
values    score
 35345 3.331550
     2 1.284752

Top anomalies
values    score
 35345 3.129754
 24562 1.766415
     2 1.338806

```

# Performace
Using the PCA model, is capable of training and testing 1 million numbers in 0.38 seconds

# Docker

To run the tool using Docker use our Docker Hub image:

```bash
docker run stratosphereips/number_anomaly_detector:latest number_anomaly_detector.py -f test-numbers.txt
```
