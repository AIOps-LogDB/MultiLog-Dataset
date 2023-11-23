# Dataset of MultiLog

## Paper Title

Multivariate Log-based Anomaly Detection for Distributed Database

## Overview

The dataset download link: (the link will be made publicly available after being accepted, now we provide some sample dataset files)

This dataset is mainly designed for cluster anomaly detection:

| No. | Anomaly                       | Cause Type | Description                                                       |
|-----|-------------------------------|------------|-------------------------------------------------------------------|
| 1   | CPU Saturation                | System     | The CPU computing resources exhaust.                               |
| 2   | IO Saturation                 | System     | The I/O bandwidth is heavily occupied.                             |
| 3   | Memory Saturation             | System     | Insufficient memory resources.                                     |
| 4   | Network Bandwidth Limited     | System     | The network bandwidth between nodes is limited.                    |
| 5   | Network Partition Arise       | System     | Network partition occurs between nodes.                            |
| 6   | Machine Down                  | System     | One server goes down when the applications are running.            |
| 7   | Accompanying Slow Query       | Database   | Excessive query load.                                              |
| 8   | Export Operations             | Database   | Backing up data to external source.                                |
| 9   | Import Operations             | Database   | Importing data from external source.                               |
| 10  | Resource-Intensive Compaction | Database   | Compaction tasks consume a substantial amount of system resources. |
| 11  | Overly Frequent Disk Flushes  | Database   | The low interval of flush operations leads to frequent disk writes.|


