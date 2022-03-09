# Newspaper Page Segmentation Example

This is an example showing the use of Mask R-CNN in segmenting newspaper pages into different components.

## Usage
```bash
usage: newspaper.py [-h] [--dataset /path/to/newspaper/dataset/]
                    [--weights /path/to/weights.h5] [--logs /path/to/logs/]
                    [--image path to image]
                    [--out-dir path to output directory]
                    [--classes List of classes. [List of classes. ...]]
                    <command>

Train Mask R-CNN to segment newspaper pages.

positional arguments:
  <command>             'train' or 'test'

optional arguments:
  -h, --help            show this help message and exit
  --dataset /path/to/newspaper/dataset/
                        Directory of the Newspaper dataset. This directory
                        should contain 'train', 'val', and 'test' folders.
  --weights /path/to/weights.h5
                        Path to weights .h5 file or 'coco'. If this option is
                        not provided, the most recently trained weights are
                        used.
  --logs /path/to/logs/
                        Logs and checkpoints directory (default=logs/)
  --image path to image
                        Image to apply the model and generate newspaper page
                        segments.
  --out-dir path to output directory
                        Output directory path to store segment images.
  --classes List of classes. [List of classes. ...]
                        Classes to test the trained model on.
```

## Train the Newspaper Model

Train a new model starting from pre-trained COCO weights
```bash
python3 newspaper.py train --dataset=/path/to/newspaper/dataset --weights=coco
```

Resume training a model that you had trained earlier
```bash
python3 newspaper.py train --dataset=/path/to/newspaper/dataset --weights=last
```

The code in `newspaper.py` is set to train for 3K steps (30 epochs of 100 steps each), and using a batch size of 2.
Update the schedule to fit your needs.

## Test the Trained Newspaper Model and Generate Segments.

Test the trained model on an image:

```bash
python3 newspaper.py test --weights=/path/to/weights/file.h5 --image=<path to file>
```

Test the trained model on an image dataset:

```bash
python3 newspaper.py test --weights=/path/to/weights/file.h5 --dataset=<path to directory containing images>
```

## Run Jupyter notebooks
Open the `inspect_newspaper_data.ipynb` or `inspect_newspaper_model.ipynb` Jupyter notebooks. You can use these notebooks to explore the dataset and run through the detection pipeline step by step.


## Newspaper Collection Annotation

We used VIA (VGG Image Annotator - version 2.0.10) to annotate sample images from the Library of Congress Chronicling America Newspaper collection.

The following are the annotation attributes (region/file attributes in VIA) that we used:

```json
{
  "region": {
    "Newspaper": {
      "type": "dropdown",
      "description": "Newspaper",
      "options": {
        "0": "background",
        "1": "ad",
        "2": "article",
        "3": "masthead",
        "4": "run_head",
        "5": "photo",
        "6": "banner"
      },
      "default_options": {
        "2": true
      }
    }
  },
  "file": {}
}
```

From this, the `background` class was not used while actually doing the annotations.
