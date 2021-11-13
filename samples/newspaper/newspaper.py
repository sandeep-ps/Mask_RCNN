"""
Mask R-CNN
Train on example newspaper dataset and test on an image or dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Modified by Sandeep Puthanveetil Satheesan
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 newspaper.py train --dataset=/path/to/newspaper/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 newspaper.py train --dataset=/path/to/newspaper/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 newspaper.py train --dataset=/path/to/newspaper/dataset --weights=imagenet

    # Test trained model on an image
    python3 newspaper.py test --weights=/path/to/weights/file.h5 --image=<path to file>

    # Test trained model on an image dataset
    python3 newspaper.py test --weights=/path/to/weights/file.h5 --dataset=<path to directory containing images>
"""

import cv2
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import warnings
from imgaug import augmenters as iaa
from mrcnn.visualize import display_images

# Ignore warnings
warnings.filterwarnings("ignore")

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class NewspaperConfig(Config):
    """Configuration for training on the Newspaper dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Newspaper"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    # Background + Ad + Article + Masthead + Runhead + Photo + Banner
    NUM_CLASSES = 1 + 6

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100  # Initial value was 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7  # Initial value was 0.9

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.01  # Initial value was 0.1
    LEARNING_MOMENTUM = 0.9

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    # TRAIN_ROIS_PER_IMAGE=20


############################################################
#  Dataset
############################################################

class NewspaperDataset(utils.Dataset):

    def load_newspaper(self, dataset_dir, subset):
        """Load a subset of the Newspaper dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have seven classes to add.
        self.add_class("newspaper", 1, "ad")
        self.add_class("newspaper", 2, "article")
        self.add_class("newspaper", 3, "masthead")
        self.add_class("newspaper", 4, "run_head")
        self.add_class("newspaper", 5, "photo")
        self.add_class("newspaper", 6, "banner")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (version 2.0.8) saves each image in the form:
        # {
        #     "filename": "lccn_97071090_1911-06-20_ed-1_seq-6.png",
        #     "size": 16680543,
        #     "regions": [
        #         {
        #             "shape_attributes": {
        #                 "name": "rect",
        #                 "x": 378,
        #                 "y": 400,
        #                 "width": 723,
        #                 "height": 2722
        #             },
        #             "region_attributes": {
        #                 "Newspaper": "2"
        #             }
        #         },
        #         {
        #             "shape_attributes": {
        #                 "name": "rect",
        #                 "x": 1117,
        #                 "y": 394,
        #                 "width": 696,
        #                 "height": 1769
        #             },
        #             "region_attributes": {
        #                 "Newspaper": "2"
        #             }
        #         }
        #     ]
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. These are stored in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                for r in a['regions']:
                    r['shape_attributes']['name'] = next((class_info['name'] for class_info in self.class_info
                                                          if class_info['id'] == int(r['region_attributes']["Newspaper"])))
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only manageable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "newspaper",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a newspaper dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "newspaper":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.rectangle((p['y'], p['x']), extent=(p['height'], p['width']))
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance.
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(p['name']) for p in info["polygons"]])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "newspaper":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = NewspaperDataset()
    dataset_train.load_newspaper(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NewspaperDataset()
    dataset_val.load_newspaper(args.dataset, "val")
    dataset_val.prepare()

    # Image augmentation
    # https://imgaug.readthedocs.io/en/latest/source/overview_of_augmenters.html
    augmentation = iaa.SomeOf((0, 3), [
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-5, 5)),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 1.0))
    ])

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,  # 30
                augmentation=augmentation,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


def generate_and_save_segments(image, result, dirname, filename, class_ids=None):

    segmented_images = []

    # Iterate through the regions and save the image segments
    for i in range(len(result['rois'])):

        x = result['rois'][i][0]
        y = result['rois'][i][1]
        width = result['rois'][i][2]
        height = result['rois'][i][3]
        class_id = result['class_ids'][i]

        # If class IDs are provided, segment only those classes.
        if class_ids is None or class_id in class_ids:
            crop_img = image[x:width, y:height]
            path = os.path.join(dirname, filename + "-seg-" + str(i) + ".png")
            cv2.imwrite(path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
            segmented_images.append(crop_img)

    return segmented_images


def detect_and_segment_single_image(model, image_path, out_dir, class_ids=None):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    # Get dir and filenames
    if out_dir is None:
        dirname = os.path.dirname(image_path)
    else:
        dirname = out_dir
        # Create output folder
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    filename = os.path.basename(image_path).split(".")[0]
    # Segment images
    segmented_images = generate_and_save_segments(image, r, dirname, filename, class_ids=class_ids)
    # Display image segments
    # TODO: Commented out for now.
    # display_images(segmented_images)


def detect_and_segment(model=None, image_path=None, dataset=None, out_dir=None, class_names=None):

    class_ids = []

    # Image
    if image_path and os.path.isfile(image_path):
        # Run model detection and generate newspaper segments
        print("Running on {}".format(args.image))
        # TODO: Fix class_ids for individual images and then uncomment below line.
        # detect_and_segment_single_image(model, image_path, out_dir, class_ids=class_ids)
    # Test dataset
    elif dataset and os.path.isdir(dataset):
        # Read dataset
        val_dataset = NewspaperDataset()
        val_dataset.load_newspaper(dataset, "test")
        val_dataset.prepare()

        print(val_dataset.map_source_class_id("newspaper.5"))
        print(val_dataset.class_names)

        for info in val_dataset.class_info:
            for class_name in class_names:
                if info["name"] == class_name:
                    class_ids.append(info["id"])

        image_filenames_list = []
        dir_path = ""
        test_dataset_path = os.path.join(dataset, "test")
        # Get filenames
        for (dir_path, dir_names, filenames) in os.walk(test_dataset_path):
            image_filenames_list.extend(filenames)
            break
        # Get filenames
        for image_filename in image_filenames_list:
            if str(image_filename).endswith(".png"):
                # Run model detection and generate newspaper segments
                print("Running on {}".format(image_filename))
                detect_and_segment_single_image(model, os.path.join(dir_path, image_filename), out_dir,
                                                class_ids=class_ids)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to segment newspaper articles.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/newspaper/dataset/",
                        help='Directory of the Newspaper dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'", default=None)
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path to image",
                        help='Image to apply the model and generate newspaper article segments.')
    parser.add_argument('--out-dir', required=False,
                        metavar="path to output directory",
                        default=None,
                        help='Output directory path to store segment images.')
    parser.add_argument('--classes', required=False,
                        metavar="List of classes.",
                        action="append", nargs="+", type=str, default=None,
                        choices=["ad", "article", "masthead", "run_head", "photo", "banner"],
                        help='Classes to test the trained model on.')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "test":
        assert args.image or args.dataset,\
               "Provide --image or --dataset to apply model and generate newspaper article segments."

    # Configurations
    if args.command == "train":
        config = NewspaperConfig()
    else:
        class InferenceConfig(NewspaperConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    if args.weights is None:
        args.weights = model.find_last()

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or Test
    if args.command == "train":
        train(model)
    elif args.command == "test":
        detect_and_segment(model, image_path=args.image, dataset=args.dataset, out_dir=args.out_dir,
                           class_names=args.classes[0])
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'test'".format(args.command))
