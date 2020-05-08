# Artificial Intelligence
#### Álvaro Martínez Fernández
#### Madhu Koirala
#### University of Tromsø - 2020

## Usage
We used Python 3.8 to build the system. You can use a version higher than 3.6 to run the scripts for this project.

The system is divided in several scripts that can be run individually. Read the following descriptions to know more about what each script does.

### main.py
This is a command-line version of the project. Running this file will make recommendations for user that is picked up randomnly and the recommendations will be shown them in the console.

### gui.py
This is the graphic user interfae version of the recommendation system. It will show a GUI with song recommendations for a user that is picked up randomnly everytime the script is run.

### plot.py
It will plot a precission-recall curve graph comparing the popularity and the collaborative filtering approaches.

### subset.py
This will create a subset and save it into the dataset folder. In the script the variable size at the beginning will determine how big the subset will be. The the dataset is retrieved from the millionsongs dataset using a http protocol.