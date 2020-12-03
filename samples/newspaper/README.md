# Newspaper Article Segment Example

This is an example showing the use of Mask RCNN in segmenting newspaper articles.

## Apply color splash using the provided weights
Apply splash effect on an image:

```bash
python3 newspaper.py splash --weights=/path/to/mask_rcnn/mask_rcnn_newspaper.h5 --image=<file name or URL>
```

## Run Jupyter notebooks
Open the `inspect_newspaper_data.ipynb` or `inspect_newspaper_model.ipynb` Jupyter notebooks. You can use these notebooks to explore the dataset and run through the detection pipeline step by step.

## Train the Newspaper model

Train a new model starting from pre-trained COCO weights
```
python3 newspaper.py train --dataset=/path/to/newspaper/dataset --weights=coco
```

Resume training a model that you had trained earlier
```
python3 newspaper.py train --dataset=/path/to/newspaper/dataset --weights=last
```

Train a new model starting from ImageNet weights
```
python3 newspaper.py train --dataset=/path/to/newspaper/dataset --weights=imagenet
```

The code in `newspaper.py` is set to train for 3K steps (30 epochs of 100 steps each), and using a batch size of 2. 
Update the schedule to fit your needs.
