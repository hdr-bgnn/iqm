Source code for "Biodiversity Image Quality Metadata Augments Convolutional Neural Network Classification of Fish Species" submitted to MTSR

```
Organization:
├── HGNN
│   ├── analyze
│   │   ├── analyze_experiments.ipynb                              Confusion matrices
│   │   ├── analyze_trial.ipynb                                    
│   │   ├── calculate_random_accuracy.ipynb                        
│   │   ├── correlation_of_features.ipynb                          
│   │   ├── flashtorch_modified.py                                 
│   │   ├── obtain_misclassified_examples.ipynb                    Extract misclassified and look at quality factors
│   │   └── plot_network.ipynb                                     
│   ├── dataPrep
│   │   ├── Segment_images.ipynb                                   
│   │   ├── copy_image_subset.ipynb                                Useful for subsets (called "suffix" for some reason)
│   │   ├── create_cleaned_metadata.ipynb                          metadata -> cleaned_metadata
│   │   ├── generate_augmented_data.ipynb                          rotate and such
│   └── train
│       ├── CNN.py                                                 Where the rubber meets the road
│       ├── CSV_processor.py                                       Parse a experiment-wide result and metadata file
│       ├── ConfigParserWriter-QualityExperiments.ipynb            Generate config for the quality experiments
│       ├── configparser_all_graded_images.ipynb                   Generate config for the whole corpus experiment
│       ├── cifar_nin.py                                           
│       ├── configParser.py                                        Parse an individual experiment config
│       ├── dataLoader.py                                          Train/test manager
│       ├── resnet_cifar.py                                        
│       └── train.py                                               Run everything
```

Install dependencies:
Using pip
```
virtualenv .
pip install -r pip_requirements.txt
```
or using Conda
```
conda create -n hgnn
conda activate hgnn
conda install conda_requirements.yaml
```

Install myhelpers:
```
pip install git+ssh://git@github.com/elhamod/myhelpers.git
```

Setup paths and parameters:

Alter the `config.default.ini` to create a `config.ini` in the HGNN directory
```
[general]
experimentsPath = /home/elhamod/HGNN/experiments/
dataPath = /home/elhamod/projects/HGNN/data
experimentName = BestModel
fine_csv_fileName = metadata.csv

[dataPrep]
image_path = INHS_cropped
output_directory_name = 52
cleaned_species_csv_fileName = cleaned_metadata.csv
species_csv_fileName = metadata.csv
numberOfImagesPerSpecies = 50

[augmented]
applyHorizontalFlip = True
applyHorizontalFlipProbability = 0.1
applyTranslation = True
applyTranslationAmount = 0.1
applyColorPCA = True
colorPCAperturbationMagnitude = 0.1
numOfAugmentedVersions = 10
cuda = 0
    

[train]
modelFinalCheckpoint = finalModel.pt
statsFileName = stats.csv
timeFileName = time.csv
epochsFileName = epochs.csv
paramsFileName = params.json


# cleaned up metadata file that has no duplicates, invalids, etc
cleaned_fine_csv_fileName = cleaned_metadata.csv

# Saved file names.
statistic_countPerFine = count_per_fine.csv
statistic_countPerFamilyAndGenis = count_per_family_genus.csv

experimentsFileName = experiments.csv

[metadatatableheaders]
fine_csv_fileName_header = fileName
fine_csv_scientificName_header = scientificName
fine_csv_Coarse_header = Genus
fine_csv_Family_header = Family

[dataloader]
testIndexFileName = testIndex.csv
valIndexFileName = valIndex.csv
trainingIndexFileName = trainingIndex.csv
paramsFileName = params.json
```

- Have your data with metadata.csv in a directory


- Create a config file using HGNN/train/ConfigParserWriter-*
- train a model using train.py. e.g.: python3 train.py --cuda=7 --name="learningRateTest" --experiments="/home/elhamod/HGNN/experiments/" --data="/data/BGNN_data"
- analyze the data using jupyter notebooks under HGNN/analyze/ .e.g Analyze trial, Analyze experiments.
- Once trained and analyzed, an experiment folder with name <experiments>/<name> will have been created, with a "models" folder that has all the trained model, a "results" folder with the analysis, and a "datasplits" folder with the indexes of images used for train/val/test.
