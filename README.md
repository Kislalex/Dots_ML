# Dots_ML
This is a simple program that trains dots to go towards the goal avoiding obsticales through different levels.
Here the simple approach to ML used.

### Prerequisites

This program runs on Python3.7.
Using the following libraries: numpy  for mathematical operations,
                               opencv for drawing 2D,
                           and shapely for computing geometric objects.

```
pip install numpy shapely opencv-python

```

### Installing

Download the .zip archiv or clone the repo

```
git clone ...
```

And run the drawing_box.py

```
./drawing_box.py
```
or
```
python3.9 drawing_box.py
```

## Running the tests

Different field are given in the folder "fields". 
Each field contains a "Goal" point, then the list of polygons representing obsticales,
then separated by '-1' the list of checkpoint's that points should pass in the correct order.
The file should end by '-1'

