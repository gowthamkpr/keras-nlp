import keras
import numpy as np
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.diffbin.diffbin_backbone import DiffBinBackbone
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.models.image_text_detector import (mask_to_polygons, 
    unclip_polygon
)



@keras_hub_export("keras_hub.models.ImageSegmenterPreprocessor")
class DiffBinPreprocessor(Preprocessor):
    """Differential Binarization scene text detection task.
     
    DiffBinImageTextDetector tasks wrap a ImagePreprocessor for preprocessing 
    input image and masks and generate polygon representation of the mask outputs
    """
    
    def __init__(
        self,
        image_converter=None,
        resize_output_mask=False,
        detection_thresh= 0.3,
        min_area=10.0,
        unclip_ratio=2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_converter = image_converter
        self.resize_output_mask = resize_output_mask
        self.detection_thresh = detection_thresh
        self.min_area = min_area
        self.unclip_ratio = unclip_ratio

    def get_preproccess(
            self,
            x,
            y=None,
            sample_weight=None
        ):
        """Preprocesses input data fby applying image conversion and rezing of 
        output masks.

        Args:
        x: Input Image or batches of Images.
        y: Target mask or batches of masks.(optional)
        sample_weight: Sample weights for the batch.(optional)

        Returns:
        Tuple of (x, y, sample_weight) after applying image conversion

        """
        
        if self.image_coverter:
            x= self.image_coverter(x)
        if y is not None and self.image_converter and self.resize_output_mask:
            y= keras.layers.Resizing(
                height=(
                    self.image_converter.image_size[0]
                    if self.image_converter.image_size
                    else None
                ),
                width=(
                    self.image_converter.image_size[1]
                    if self.image_converter.image_size
                    else None
                ),
                crop_to_aspect_ratio=self.image_converter.crop_to_aspect_ratio,
                interpolation="nearest",
                data_format=self.image_converter.data_format,
                dtype=self.dtype_policy,
                name='mask_resizing'
            )(y)
        return keras.utils.unpack_x_y_sample_weight(x, y, sample_weight)
    
    def get_postprocess(
            self, 
            masks, 
            contour_finder="simple"):
        """Converts the mask output of a text detector to polygon coordinates.

        Args:
            masks: Segmentation masks (3D batch of masks).
            contour_finder: Determines the method for contour finding. Possible
                values are "simple", which detects connected regions by walking
                the image, and "opencv", which uses OpenCV's contour finder if
                available. Defaults to "simple".

        Returns:
            List-of-list-of-lists. A list of polygons for each batch element,
            where each polygon is represented as a list of (x, y) points.
        """
        if not isinstance(masks,np.ndarray):
            masks= keras.ops.convert_to_numpy(masks)
        masks= masks > self.detection_thresh
        polygons=[]
        for mask in masks:
            mask_polygons = mask_to_polygons(
                mask, min_area=self.min_area, contour_finder=contour_finder
            )
            mask_polygons = [
                unclip_polygon(polygon, self.unclip_ratio)
                for polygon in mask_polygons
            ]
            polygons.append(mask_polygons)
        return polygons
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "detection_thresh": self.detection_thresh,
                "min_area": self.min_area,
                "unclip_ratio": self.unclip_ratio,
            }
        )
        if self.image_converter is not None:
            config["image_converter"] = self.image_converter.get_config()
        return config