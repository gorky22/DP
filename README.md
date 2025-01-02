## Overview

This project is Diploma Thesis for matching debris fragments detected on AMOS camera systems (https://opac.crzp.sk/?fn=detailBiblioFormChildEGLP&sid=AA0D9432EF6DE826C913184F11D3&seo=CRZP-detail-kniha).

## Setup and Installation

Instructions on setting up and running the project:

1. Clone the repository.
2. Install required dependencies (list any necessary libraries or packages) from `requirements.txt`.
3. Run the main script to run pipeline (xmls and yamls\_ are folders containing data from AMOS camera system)

## Folder Structure

/
| get_data.py # Script to fetch or generate data needed for processing.
| main.py # Main entry point of the project.
| match.py # Script for matching data patterns or positions.
| train.py # Script to train models based on the processed data.

|── /data_simulator # Contains scripts for simulating and creating synthetic data.
| | fragment_simulator.py # Simulates data fragments.
| | synthetic_data_creator.py # Creates synthetic data sets.

|── /position_matcher # Contains scripts for position matching algorithms.
| | meteor.py # Handles meteor-related position data.
| | xmldict.py # Utility script for handling XML dictionary structures.

|── /superglue # Framework for model management.
| |── /models
| | | superglue.py # Core script for model operations.
| |── /weights
| | | train.py # Script for training the models with new data.

| | get_data.py # General script generate synthetic data.
| | match.py # Script to perform data matching operations.
| | main.py # Main for managing pipeline.

markdown
Copy code

## How to Use

- Run `main.py` To start pipeline process with best model we have if else add new to folder weights. Example 'python3 main.py yamls*/HK xmls/HK yamls*/MK xmls/MK'
- Use `train.py` in the `/weights` file to train models using the provided data. Example 'python3 main.py'
- Use `get_data.py` in the `/weights` file to generate synthetic data. Example 'python3 get_data.py'
# diploma_thesis
