# Smart Trash Project

## Database
The database used is the famous [Garbage Classification](https://www.kaggle.com/asdasdasasdas/garbage-classification) dataset found on Kaggle. It was manually separated into Training and Validation images in separate folders, which can be found above in the repo.

## Notes
The [train.py](https://github.com/A223D/smart-trash/blob/master/train.py) file was run on a GPU runtime on Google Colaboratory to speed up training. The .h5 file was exported and transferred to a Raspberry Pi 3B+, which does the detection.

[cv_oldest.py](https://github.com/A223D/smart-trash/blob/master/cv_oldest.py) was the first version of our program. This version tried to classify all 6 types of garbage(paper, plastic, metal, glass, cardboard, and trash). However, the Raspberry Pi camera was too poor to classify all of them. Additionally, the white text on the frame was difficult to see in some background settings. It also did not classify materials into biodegradable and non-biodegradable ones. There is also no LED interface.

[cv_old.py](https://github.com/A223D/smart-trash/blob/master/cv_old.py) was the second version. This one fixed the text, by adding a black outline around the letters. However, it still tried to classify all types of garbage and did not classify them into biodegradable and non-biodegradable. There is still no LED interface.

[cv_latest.py](https://github.com/A223D/smart-trash/blob/master/cv_latest.py) is the final iteration of the code. After testing, we determined that the Raspberry Pi camera sensed paper, metal, and plastic the best. We also added a simple filter that classifies paper, metal, and plastic into biodegradable and non-biodegradable materials. We also added 2 LEDs which indicate which hole to put the trash in (bio/non-bio).
