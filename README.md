# Self-Imputing Deep Multitask Sequence Model

This repository contains the source code and data for the paper:
> [**A Self-Imputing Deep Multitask Sequence Model for Traffic Disruption Detection in Extreme Conditions**](https://ieeexplore.ieee.org/abstract/document/11174266)


## Overview

This project introduces SI-DMSeq (Self-Imputing Deep Multitask Sequence), a multitask sequence-to-sequence learning model designed to address the challenges of anomaly detection and data incompleteness in urban traffic networks. 

Urban traffic networks are highly vulnerable during extreme events, where disruptions can quickly propagate and cause severe congestion. Timely and accurate detection of traffic anomalies is critical for minimizing impacts, yet existing methods often fail due to limited anomaly detection capabilities and challenges with incomplete data.

SI-DMSeq addresses these challenges in a unified framework by jointly performing traffic data imputation and anomaly detection. It not only completes missing traffic records but also detects disruptions at both the network and local scales, enabling dual-level insights into system-wide impacts and localized effects.

Key features of the SI-DMSeq model include:

- **Unified multitask framework**: simultaneously performs traffic data imputation and anomaly detection.  
- **Dual-level anomaly detection**: identifies disruptions at both the network and local (zone) levels through an autoencoder-based approach.
- **Iterative self-imputation**: refines missing values during training using the autoencoder’s reconstructions instead of relying on external preprocessing.  
- **Future Predictor module**: ensures continuous short-term forecasting and fills missing values during inference.  
- **Dynamic loss function**: adaptively balances reconstruction, prediction, and imputation to improve learning over time.  
- **Statistical control chart** + **scoring method**: highlights large-scale disruptions and pinpoints the most affected regions.  


## Requirements

Install the required dependencies with:

```bash
pip install -r requirements.txt