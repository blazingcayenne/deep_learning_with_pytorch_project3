# Project Approach

In the last project I created a framework by which I could easily create, run, and analyze experiments. I choose this approach in order to learn how to program in Python. In this project I will use an existing framework to reduce my development effort.

After reading the article **[RetinaNet: Custom Object Detection training with 5 lines of code](https://laptrinhx.com/retinanet-custom-object-detection-training-with-5-lines-of-code-1882442374/)**, I started prototyping with [Monk](https://github.com/Tessellate-Imaging/Monk_Object_Detection)'s [RetinaNet](https://www.paperswithcode.com/method/retinanet) implementation (https://www.paperswithcode.com/method/retinanet), which is built upon [PyTorch RetinaNet](https://github.com/yhenon/pytorch-retinanet). I abandoned this framework after realizing the following.

* Monk's documentation was very limited.
* Monk's RetinaNet implementation was not actively maintained and had many outstanting issues.
* Substantial refactoring and development were required to meet the project's requirements.

I ultimately decided to use [FaceBook AI](https://ai.facebook.com/)'s [Detectron2](https://github.com/facebookresearch/detectron2). Its introductory blog post, [Detectron2: A PyTorch-based modular object detection library](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/), describes this framework as follows.

> Detectron2 is a ground-up rewrite of Detectron that started with 
[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). The platform is now implemented in 
[PyTorch](https://pytorch.org/). With a new, more modular design, Detectron2 is flexible and extensible, and able to provide fast training on single or multiple GPU servers. Detectron2 includes high-quality implementations of state-of-the-art object detection algorithms, including [DensePose](http://densepose.org/), [panoptic feature pyramid networks](https://ai.facebook.com/blog/improving-scene-understanding-through-panoptic-segmentation/), and numerous variants of the pioneering [Mask R-CNN](https://research.fb.com/publications/mask-r-cnn/) model family also developed by FAIR. Its extensible design makes it easy to implement cutting-edge research projects without having to fork the [entire codebase](https://github.com/facebookresearch/detectron2).

This project fine-tunes pretrained [Detectron2 RetinaNet](https://detectron2.readthedocs.io/en/latest/modules/modeling.html?highlight=retinanet#detectron2.modeling.RetinaNet) models using the class supplied _Vehicle Registration Plate_ dataset. For brevity, a vehicle registration plate is referred to as a license plate. RetinaNet is introducted in the **[Focal Loss for Dense Object Detection](https://www.paperswithcode.com/method/retinanet)** article and its architecture is depicted below.

![RetinaNet network architecture](https://www.paperswithcode.com/media/methods/Screen_Shot_2020-06-07_at_4.22.37_PM.png)

RetinaNet is a one-stage object detection model that utilizes a focal loss function to address class imbalance during training. Focal loss applies a modulating term to the cross entropy loss in order to focus learning on hard negative examples. The impact of this modulating term is depicted for various value of ɣ in the plots below.

![Focal loss plot](https://www.starlg.cn/2019/01/10/Focal-Loss/focalLoss.png)

Detectron2 provides trainers, inferencers, model evaluators, and visualization tools. Nevertheless, I decided to implement my own versions of these components for the following reasons.

* To understand the implementation details involved in object detection.
* To understand how validation loss correlates with object detection evaluation metrics, e.g., mAP.
* To learn how to annotate images in Python on a simply problem, drawing labeled bounding boxes.
* The output of the Detectron2's components is rather untidy.

**Note:** Since the Detectron2 framework had other pretrained object detector models, this project compared RetinaNet-50 and RetinaNet-101 to faster R-CNN-X101.

# Conclusions

This project fine-tuned a pretrained RetinaNet model with the Feature Pyramid Network backbone on top of feedforward ResNet-50 and ResNet-101 architectures. For comparision, it also fine-tuned a faster R-CNN on top of a feedforward ResNext-101 architecture. The following table summarizes their mean average precision (mAP) with and without data augmentation (DA). The impact of Detectron2 configuration-based data augmentation was minimal.

|Model|mAP w/o DA|mAP w/ DA|
|:---|:---:|:---:|
|RetinaNet-50|0.613|0.615|
|RetinaNet-101|0.617|0.625|
|Faster R-CNN-X101|0.602|†|

† The last experiment, BAC-FasterRCNNX101-AUG, crashed after training for approximately 9.5 hours. Because it does not appear that this experiment will produce a model with the highest mAP, training was abandoned rather than implement code to restart the model from its last checkpoint.

Regarding frameworks, the upfront time investigating them is well spent before selecting one for a project. Checking whether the framework is actively maintained and rapid prototyping before commitment is prudent. Lastly, avoid frameworks that lack high-quality documentation.

I am impressed with FaceBook AI's Detectron2 framework and would use it again on another project. It has the following benefits.

* Actively maintained.
* High quality documentation.
* Well commented, well structured source code.
* There are many articles and blog posts on inferencing and training with it.
* There are several public Jupyter notebook examples.

# Future Investigation

The following areas warrant future exploration.

* The use of different, i.e., non-default, PyTorch and Detectron2's optimizers and learning rate schedulers.
* Comparison of non-RetinaNet object detection models (see [Detectron2's Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#coco-object-detection-baselines)).
* The use of additional Detectron2 data augmentation to mitigate overfitting via a custom [Dataset Mapper](https://detectron2.readthedocs.io/en/latest/modules/data.html?highlight=DatasetMapper#detectron2.data.DatasetMapper) or integration of [Albumentations](https://albumentations.ai/) data augmentation transforms, e.g., [denilv/detectron2-helpers](https://github.com/denilv/detectron2-helpers/blob/master/dummy_albu_mapper.py).
* The RetinaNet models detected the large, but not the small partially occluded, license plate in the first test image. The fully trained Faster R-CNN model detected the small partially occluded, but not the large, license plate. Furthermore, Faster R-CNN returned more confident predictions than RetinaNet. Hence, futher exploration between these models is warranted.
* Annotate other YouTube videos, e.g., [Driving Downtwon - New York City 4K - USA](https://youtu.be/7HaJArMDKgI), [Driving Downtown - Chicago Main Street 4K - USA](https://youtu.be/gEKwMXGFimk).

# External Project Links

For convenience, links to this project's TensorBoard logs and YouTube videos are copied below.

* [TensorBoard Logs](https://tensorboard.dev/experiment/IG8BBp0BSK6g7fOpiaNj3Q/#scalars&_smoothingWeight=0)
* [RetinaNet101 Video](https://youtu.be/oMURlILCDqo), [Faster R-CNN X101 Video](https://youtu.be/741phKYWYVM)