# GoogleLandmarkRetrieval

This is my solution for [Google Landmark Retrieval Challenge](https://www.kaggle.com/c/landmark-retrieval-challenge/)

Description is available at [here]()

### Requirement
- python
- keras
- cv2
- faiss
- download image data from [here](https://www.kaggle.com/c/landmark-retrieval-challenge/discussion/56194)
and place them in `input/index/` and `input/query`  
- place `sample_submission.csv` in `input/`

### How to use
- run `extract_feature_vgg.py`  
  extract feature at dir `output/`  
  you can change a layer to extract features by change line 57   
- run `output_submission.py`  
  calcurate distances between features and output submission file at dir `output/`  
  you can change a layer to extract features by editing line 7  
  you can change a model to extract features by editing line 8  
- run `train_rotnet.py`  
  train [RotNet](https://openreview.net/forum?id=S1v4N2l0-)
- run `extract_feature_rotnet.py`  
  extract feature at dir `output/`  
  you can change a layer to extract features by change line 78 
  
