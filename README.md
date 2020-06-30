This repo is for Fish classification.

Organization:
├── HGNN
│   ├── analyze
│   │   ├── analyze_experiment.ipynb                               What is this?
│   │   ├── analyze_experiments.ipynb                              What is this?
│   │   ├── analyze_trial.ipynb                                    What is this?
│   │   ├── calculate_random_accuracy.ipynb                        What is this?
│   │   ├── correlation_of_features.ipynb                          What is this?
│   │   ├── flashtorch_modified.py                                 What is this?
│   │   ├── obtain_misclassified_examples.ipynb                    What is this?
│   │   └── plot_network.ipynb                                     What is this?
│   ├── dataPrep
│   │   ├── Segment_images.ipynb                                   What is this?
│   │   ├── copy_image_subset.ipynb                                What is this?
│   │   ├── create_cleaned_metadata.ipynb                          What is this?
│   │   ├── experiment-test-dataloader2-full_INHS.ipynb            What is this?
│   │   ├── generate_augmented_data.ipynb                          What is this?
│   │   └── imageDownloadScript.sh                                 What is this?
│   └── train
│       ├── CNN.py                                                 What is this?
│       ├── CSV_processor.py                                       What is this?
│       ├── ConfigParserWriter-GridExperiments.ipynb               What is this?
│       ├── ConfigParserWriter-HyperParameterExperiments.ipynb     What is this?
│       ├── ConfigParserWriter-SelectExperiments.ipynb             What is this?
│       ├── cifar_nin.py                                           What is this?
│       ├── configParser.py                                        What is this?
│       ├── dataLoader.py                                          What is this?
│       ├── resnet_cifar.py                                        What is this?
│       └── train.py                                               What is this?

To start:

Complete the `config.default.ini` to create a `config.ini`
```
[analyze]
experimentsPath = /home/elhamod/HGNN/experiments/
dataPath = /data/BGNN_data
experimentName= BestModelForJeremy

[dataPrep]
User = hg

[train]
modelFinalCheckpoint = 'finalModel.pt'
statsFileName = "stats.csv"
timeFileName = "time.csv"
epochsFileName = "epochs.csv"
paramsFileName="params.json"


[csvpreprocessor]
# metadata file provided by dataset.
fine_csv_fileName = "metadata.csv"
# cleaned up metadata file that has no duplicates, invalids...etc
cleaned_fine_csv_fileName = "cleaned_metadata.csv"

# Saved file names.
statistic_countPerFine="count_per_fine.csv"
statistic_countPerFamilyAndGenis="count_per_family_genus.csv"

# metadata table headers.
fine_csv_fileName_header = "fileName"
fine_csv_scientificName_header = "scientificName"
fine_csv_Coarse_header = "Genus"
fine_csv_Family_header = "Family"

[dataloader]
testIndexFileName = "testIndex.csv"
valIndexFileName = "valIndex.csv"
trainingIndexFileName = "trainingIndex.csv"
paramsFileName="params.json"

[train]
experimentsFileName = "experiments.csv"
```

- Have your data with metadata.csv in a directory


- Create a config file using HGNN/train/ConfigParserWriter-*
- train a model using train.py. e.g.: python3 train.py --cuda=7 --name="learningRateTest" --experiments="/home/elhamod/HGNN/experiments/" --data="/data/BGNN_data"
- analyze the data using jupyter notebooks under HGNN/analyze/ .e.g Analyze trial, Analyze experiments.
- Once trained and analyzed, an experiment folder with name <experiments>/<name> will have been created, with a "models" folder that has all the trained model, a "results" folder with the analysis, and a "datasplits" folder with the indexes of images used for train/val/test.
