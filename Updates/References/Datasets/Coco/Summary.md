### What is Coco dataset

Common Object in Context(Coco) is a dataset published by MS for solving computer vision problems. It is used as a benchmark to validate model.

##### Classes identified by Coco
Coco can classify about 80 objects. 
'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis','snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'

##### Annotations
1. Object Detection: Bounding boxes and per-instance segmentation masks with 80 object categories,
2. Captioning: Natural language descriptions of the images (see MS COCO Captions),
3. Keypoints detection: Contains more than 200,000 images and 250,000 person instances labeled with keypoints (17 possible keypoints, such as left eye, nose, right hip, right ankle),
stuff image segmentation – per-pixel segmentation masks with 91 stuff categories, such as grass, wall, sky (see MS COCO Stuff),
4. Panoptic: Contains full scene segmentation, with 80 thing categories (such as person, bicycle, elephant) and a subset of 91 stuff categories (grass, sky, road),
5. Dense pose: Contains more than 39,000 images and 56,000 person instances labeled with DensePose annotations – each labeled person is annotated with an instance id and a mapping between image pixels that belong to that person body and a template 3D model. The annotations are publicly available only for training and validation images.

##### Dataset Format

###### Basic Structure
'''
{
    "info":{
        "description": "Coco Dataset",
        "url":"htto://cocodataset.org",
        "version":"1.0",
        "year":"2007",
        "contributor":"COCOConsortium",
        "date_created": "2017/01/01"
    },
    "licenses":{
        [
            {
                "url":"license-url",
                "id":1 //License Id
                "name":"licensename"
            },
            {
                ...
            }
        ]
    },
    "images":{
        "license": 1,
        "file_name": "my_image.jpg",
        "coco_url:" "http://my_image.jpg",
        "height": 360,
        "width": 640,
        "date_captured:"2007/01/01"
        "flickr_url":"http://url",
        id: 1
    },
    "categories":{
        [
            {
                "supercategory": "person",
                "id": 1,
                "name":"person"
            },

        ]
    },
    "annotations":{
        "segmentation":
        [[
            200.0,
            350,
            220
        ]],
        "area":26677,
        "iscrowd":0,
        "image_id":1,
        "bbox":
        [
            100,
            100,
            80,
            80
        ],
        "category_id":1,
        id: 1

    }
}

'''



