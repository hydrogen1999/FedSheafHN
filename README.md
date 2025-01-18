# FedSheafHN: Federated Learning with Sheaf HyperNetworks
FedSheafHN is a federated learning framework that integrates sheaf-based models with hypernetworks to enhance collaborative learning across multiple clients. This repository provides a comprehensive implementation, including data partitioning, model training, and evaluation pipelines.
## Table of Contents

- [Overview](#tổng-quan)
- [Directory Structure](#cấu-trúc-thư-mục)
- [Installation](#cài-đặt)
- [Configuration](#cấu-hình)
- [Data Partitioning](#chia-phân-vùng-dữ-liệu)
- [Running the Pipeline](#chạy-pipeline)
- [Components](#các-thành-phần)
  - [Data Generators](#trình-tạo-dữ-liệu)
  - [FL Module](#module-học-liên-kết-phân-tán-fl)
  - [Models](#mô-hình)
  - [Parameter Generators](#trình-tạo-tham-số)
  - [Aggregators](#aggregators)
  - [Utilities](#tiện-ích)
- [Logging and Checkpointing](#ghi-nhận-và-checkpointing)
- [Customization](#tùy-biến)

## Overview

FedSheafHN leverages sheaf theory to model complex relationships within graph-structured data in a federated learning setting. By incorporating hypernetworks, the framework dynamically generates model parameters tailored to each client's local data distribution, fostering more robust and personalized learning across diverse datasets.

## Directory Structure
FedSheafHN/ ├── aggregator/ │ ├── base.py │ ├── factory.py │ └── sheaf.py ├── data/ │ ├── loader.py │ └── generators.py ├── fl/ │ ├── client.py │ ├── main.py │ ├── manager.py │ └── server.py ├── models/ │ ├── gcn.py │ └── neural_sheaf/ │ ├── lib/ │ │ └── laplace.py │ └── server/ │ ├── disc_models.py │ ├── laplacian_builders.py │ ├── orthogonal.py │ ├── sheaf_base.py │ └── sheaf_models.py ├── param_generator/ │ ├── base.py │ ├── factory.py │ └── hypernetwork.py ├── utils/ │ ├── forked_pdb.py │ ├── logger.py │ └── torch_utils.py ├── script/ │ └── disjoint.sh ├── configs/ │ ├── disjoint.yaml │ └── overlapping.yaml ├── logs/ │ └── ... (auto-generated) ├── checkpoints/ │ └── ... (auto-generated) ├── main.py ├── requirements.txt └── README.md

### Description

- **aggregator/**: Contains aggregator classes responsible for combining updates from clients.
- **data/**: Handles data loading and partitioning
  - `loader.py`: Loads and preprocesses data partitions for clients.
  - `generators.py`: Generates data partitions (disjoint or overlapping).
- **fl/**: Core federated learning components.
  - `client.py`: Defines client-side training logic.
  - `server.py`: Defines server-side aggregation and coordination logic.
  - `manager.py`: Manages the overall federated learning process.
  - `main.py`: Entry point for the federated learning process.
- **models/**: Contains model architectures
  - `gcn.py`: Graph Convolutional Network implementation.
  - `neural_sheaf/`: Sheaf-based model implementations.
- **param_generator/**: Generates model parameters using hypernetworks
  - `hypernetwork.py`: Hypernetwork implementation for parameter
  - `factory.py` and `base.py`: Facilitate the creation and management of different parameter generators.
- **utils/**: Utility scripts and helper functions.
  - `logger.py`: Logging utilities.
  - `torch_utils.py`: PyTorch-related utility functions.
  - `forked_pdb.py`: Debugging utilities.
- **script/**: Shell scripts to execute various tasks
  - `disjoint.sh`: Script to run the pipeline with disjoint data partitions.
- **configs/**: Configuration files in YAML format.
  - `default.yaml`: Configuration for disjoint data partitioning.
- **logs/**: Directory to store log files (auto-generated).
- **checkpoints/**: Directory to store model checkpoints (auto-generated).
- **main.py**: Main entry point script to initiate data partitioning and federated learning.
- **requirements.txt**: Python dependencies.
- **README.md**: Project documentation.

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA (optional, for GPU support)

### Steps

1. **Clone the Repository**

   ```
   git clone https://github.com/yourusername/FedSheafHN.git
   cd FedSheafHN
   ```

2. **Create a Virtual Environment**

  ```
  python3 -m venv venv
  source venv/bin/activate
  ```
3. **Install Dependencies**

  ```
  pip install -r requirements.txt
  ```
