import keras
from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.differential_binarization.differential_binarization_backbone import (
    DifferentialBinarizationBackbone,
)
from keras_hub.src.models.differential_binarization.differential_binarization_preprocessor import (
    DifferentialBinarizationPreprocessor,
)
from keras_hub.src.models.differential_binarization.losses import DBLoss
from keras_hub.src.models.image_segmenter import ImageSegmenter


@keras_hub_export("keras_hub.models.DifferentialBinarizationOCR")
class DifferentialBinarizationOCR(ImageSegmenter):
    """
    A Keras model implementing the Differential Binarization
    architecture for scene text detection, described in
    [Real-time Scene Text Detection with Differentiable Binarization](
    https://arxiv.org/abs/1911.08947).

    Args:
        backbone: A `keras_hub.models.DifferentialBinarizationBackbone`
            instance.
        preprocessor: `None`, a `keras_hub.models.Preprocessor` instance,
            a `keras.Layer` instance, or a callable. If `None` no preprocessing
            will be applied to the inputs.

    Examples:
    ```python
    input_data = np.ones(shape=(8, 224, 224, 3))

    image_encoder = keras_hub.models.ResNetBackbone.from_preset(
        "resnet_vd_50_imagenet"
    )
    backbone = keras_hub.models.DifferentialBinarizationBackbone(image_encoder)
    detector = keras_hub.models.DifferentialBinarizationOCR(
        backbone=backbone
    )

    detector(input_data)
    ```
    """

    backbone_cls = DifferentialBinarizationBackbone
    preprocessor_cls = DifferentialBinarizationPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):

        # === Functional Model ===
        inputs = backbone.input
        x = backbone(inputs)
        probability_maps = x["probability_maps"]
        threshold_maps = x["threshold_maps"]
        binary_maps = step_function(probability_maps, threshold_maps)
        outputs = layers.Concatenate(axis=-1)(
            [probability_maps, threshold_maps, binary_maps]
        )

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        # === Config ===
        self.backbone = backbone
        self.preprocessor = preprocessor

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        **kwargs,
    ):
        """Configures the `DifferentialBinarizationOCR` task for training.

        `DifferentialBinarizationOCR` extends the default compilation signature
        of `keras.Model.compile` with defaults for `optimizer` and `loss`. To
        override these defaults, pass any value to these arguments during
        compilation.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses the default
                optimizer for `DifferentialBinarizationOCR`. See
                `keras.Model.compile` and `keras.optimizers` for more info on
                possible `optimizer` values.
            loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`, in which case the default loss
                computation of `DifferentialBinarizationOCR` will be applied.
                See `keras.Model.compile` and `keras.losses` for more info on
                possible `loss` values.
            **kwargs: See `keras.Model.compile` for a full list of arguments
                supported by the compile method.
        """
        if optimizer == "auto":
            optimizer = keras.optimizers.SGD(
                learning_rate=0.007, weight_decay=0.0001, momentum=0.9
            )
        if loss == "auto":
            loss = DBLoss()
        super().compile(
            optimizer=optimizer,
            loss=loss,
            **kwargs,
        )


def step_function(x, y, k=50.0):
    return 1.0 / (1.0 + keras.ops.exp(-k * (x - y)))
