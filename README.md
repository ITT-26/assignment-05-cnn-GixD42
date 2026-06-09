[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/cMaQVOgt)


# 0 Requirements and installation

- Python (3.13 used to code this)
- Install requirements using pip install -r requirements.txt
- The subset should be put in the data folder. This folder should contain: _annotations and the folders for every condition

# 1 Exploring Hyperparameters

Basically everything is explained in the notebook, which was given to us for the assignment.<br>
Some code from the course was used too.

# 2 Gathering a Dataset

All pictures and annotations are similar to the dataset.
The notebook is mostly reused and rearranged code.
The Confusion Matrix was saved and created in this notebook

# 3 Gesture-controlled Camera App

For creating the model the file model_trainer.py was created.<br>
For all the constants used in this file and the camera application the file constants.py was created.<br>
The file camera_app.py is the actual application.<br><br>
In the constants file you can change the gestures you want to train the model on and other parameters for training as well as the Path to the model<br>
UI Color and other specifications for the application can also be changed in this file.<br>
Some constants like MIN_HAND_AREA_RATIO and DIFFERENCE_THRESHOLD can impact how well the gestures get detected.<br><br>
For a robust user experience I decided to let the user hold a gesture for an in constants defined time[s] before any functionality is executed.<br>
I also decided to create an area for the gesture to be registered in because of movement in the background and contour detection.<br>
For optimal use I suggest trying to have a background with some contrast to your skin color in this specific area.<br>
The background is captured for 2 seconds so changes in this area are noticable.<br>
The detection might not work as well for every setup so I decided to make a parameter RIGHT_HAND in the constants file. If it's true the input area should be around your right hand and on your left hand if it's false.

## Startup

start the program by running

```
python camera_app.py --time time_for_selfie_countdown --path path_to_selfie_folder
```

for both parameters a standard version is given so it should run well, even if you don't use them. The folder will be created if it doesn't exist already.

## Controls

- 'R'-key recalibreates the background -> try to make a good background in the input area
- 'Q'-key closes the application
- 'like'-gesture starts the selfie countdown
- 'stop'-gesture en-/disables the sepia filter
- 'dislike'-gesture en-/disables the rain animation
