# Structure-Texture-Decomposition-GS-LPR
source code for the paper "Adaptive parameter selection for gradient sparse + low patch-rank recovery: application to image decomposition"
(Presented at EUCIPCO2024)

The project is organised as follows:
* **images** -> Folder containing some test images.
* **output** -> Folder containing some results of the method we present of structure-texture decomposition.
* **src** -> Folder containing the source code of the project.

## 1- Installation and running the code
Once the project has been cloned/downloaded, install the necessary python libraries via the command```pip install -r requirements.txt```.

There are two ways to run the code:
1. Modify the ```image_file_path``` variable in the main.py file and run the command ```python ./src/main.py``` (or run the code in your favorite IDE).
2. Run the command ```python ./src/main.py /path/to/your/image/my_image.ext```, e.g ```python ./src/main.py ./images/Barbara.tif```

In both cases, the resulting decomposition should appear in the output folder.
