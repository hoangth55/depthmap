# Update depthmap from open project

1. Clone this repository:
```
https://github.com/hoangth55/depthmap
```

2. Install the dependencies:
```
conda create -n 'name' pip python=3.6
conda activate 'name'
pip install opencv-python==3.4.0.14
pip install -r requirements.txt 
```


3. Download these two files: 
https://drive.google.com/file/d/1MRJqQ8I1--x52wO54OZTbjcWeHnNNUea/view?usp=sharing                            
https://drive.google.com/file/d/1XT2j_c0e3FevuD0-_KqAF0Ry-2RJpyNu/view?usp=drive_link
                             
   and move them to **models** folder.


4. Rename your input image as **input.jpg** and place it in the main folder.


5. Run **predict.py** file with command: 
 
```
python predict.py models/NYU_FCRN.ckpt input.jpg
```


6. It will save two depth maps as **final.jpg** and **finalgray.jpg** in the main folder.




> Please fork this repository, if you want to use this code and give it a star.
