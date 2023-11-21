# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import json
import os

from keras_nlp.backend import keras

try:
    import kagglehub
except ImportError:
    kagglehub = None

KAGGLE_PREFIX = "kaggle://"
TOKENIZER_ASSET_DIR = "assets/tokenizer"


def get_file(preset, path):
    """Download a preset file in necessary and return the local path."""
    if preset.startswith(KAGGLE_PREFIX):
        kaggle_handle = preset.removeprefix(KAGGLE_PREFIX)
        if kagglehub is None:
            raise ImportError(
                "`from_preset()` requires the `kagglehub` package. "
                "Please install with `pip install kagglehub`."
            )
        if len(kaggle_handle.split("/")) not in (4, 5):
            raise ValueError(
                "Unexpected kaggle preset handle. Kaggle model handles should have "
                "the form kaggle://{org}/{model}/keras/{variant}[/{version}]. For "
                "example, kaggle://keras-nlp/albert/keras/bert_base_en_uncased."
            )
        return kagglehub.model_download(kaggle_handle, path)
    return os.path.join(preset, path)


def get_tokenizer(layer):
    """Get the tokenizer from any KerasNLP model or layer."""
    # Avoid circular import.
    from keras_nlp.tokenizers.tokenizer import Tokenizer

    if isinstance(layer, Tokenizer):
        return layer
    if hasattr(layer, "tokenizer"):
        return layer.tokenizer
    if hasattr(layer, "preprocessor"):
        return getattr(layer.preprocessor, "tokenizer", None)
    return None


def recursive_pop(config, key):
    """Remove a key from a nested config object"""
    config.pop(key, None)
    for value in config.values():
        if isinstance(value, dict):
            recursive_pop(value, key)


def save_to_preset(
    layer,
    preset,
    save_weights=True,
    config_filename="config.json",
    weights_filename="model.weights.h5",
):
    """Save a KerasNLP layer to a preset directory."""
    os.makedirs(preset, exist_ok=True)

    # Save tokenizers assets.
    tokenizer = get_tokenizer(layer)
    assets = []
    if tokenizer:
        asset_dir = os.path.join(preset, TOKENIZER_ASSET_DIR)
        os.makedirs(asset_dir, exist_ok=True)
        tokenizer.save_assets(asset_dir)
        for asset_path in os.listdir(asset_dir):
            assets.append(os.path.join(TOKENIZER_ASSET_DIR, asset_path))

    # Optionally save weights.
    save_weights = save_weights and hasattr(layer, "save_weights")
    if save_weights:
        weights_path = os.path.join(preset, weights_filename)
        layer.save_weights(weights_path)

    # Save a serialized Keras object.
    config_path = os.path.join(preset, config_filename)
    config = keras.saving.serialize_keras_object(layer)
    # Include references to weights and assets.
    config["assets"] = assets
    config["weights"] = weights_filename if save_weights else None
    recursive_pop(config, "config_config")
    recursive_pop(config, "build_config")
    with open(config_path, "w") as config_file:
        config_file.write(json.dumps(config, indent=4))

    # Save any associated metadata.
    metadata = {
        # TODO: save keras version and keras-nlp version.
        "date_saved": datetime.datetime.now().strftime("%Y-%m-%d@%H:%M:%S"),
    }
    metadata_path = os.path.join(preset, "metadata.json")
    with open(metadata_path, "w") as metadata_file:
        metadata_file.write(json.dumps(metadata, indent=4))


def load_from_preset(
    preset,
    load_weights=True,
    config_file="config.json",
    config_overrides={},
):
    """Load a KerasNLP layer to a preset directory."""
    # Load a serialized Keras object.
    config_path = get_file(preset, config_file)
    with open(config_path) as config_file:
        config = json.load(config_file)
    config["config"] = {**config["config"], **config_overrides}
    layer = keras.saving.deserialize_keras_object(config)

    # Load any assets for our tokenizers.
    tokenizer = get_tokenizer(layer)
    if tokenizer and config["assets"]:
        for asset in config["assets"]:
            get_file(preset, asset)
        config_dir = os.path.dirname(config_path)
        asset_dir = os.path.join(config_dir, TOKENIZER_ASSET_DIR)
        tokenizer.load_assets(asset_dir)

    # Optionally load weights.
    load_weights = load_weights and config["weights"]
    if load_weights:
        weights_path = get_file(preset, config["weights"])
        layer.load_weights(weights_path)

    return layer


def check_preset_class(
    preset,
    classes,
    config_file="config.json",
):
    """Validate a preset is being loaded on the correct class."""
    config_path = get_file(preset, config_file)
    with open(config_path) as config_file:
        config = json.load(config_file)
    cls = keras.saving.get_registered_object(config["registered_name"])
    if not isinstance(classes, (tuple, list)):
        classes = (classes,)
    if cls not in classes:
        raise ValueError(
            f"Unexpected class in preset `'{preset}'`. "
            "When calling `from_preset()` on a class object, the preset class "
            f"much match allowed classes. Allowed classes are `{classes}`. "
            f"Received: `{cls}`."
        )
    return cls