"""Keras model construction.

These need TensorFlow, so they are skipped when it is not installed. The rest of
the suite stays dependency-light and runs in CI on every push; this file is the
opt-in half you run locally before touching the architecture.
"""

import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")

from roadvision.darknet import DETECTION_LAYERS, build_yolov3  # noqa: E402
from roadvision.models import build_cnn, build_perceptron  # noqa: E402


class TestClassifiers:
    def test_perceptron_shapes(self):
        model = build_perceptron()
        assert model.input_shape == (None, 32, 32, 3)
        assert model.output_shape == (None, 3)

    def test_cnn_shapes(self):
        model = build_cnn()
        assert model.input_shape == (None, 32, 32, 3)
        assert model.output_shape == (None, 3)

    def test_outputs_are_a_probability_distribution(self):
        import numpy as np

        for build in (build_perceptron, build_cnn):
            probs = build().predict(np.zeros((4, 32, 32, 3)), verbose=0)
            assert probs.shape == (4, 3)
            assert np.allclose(probs.sum(axis=1), 1.0)   # softmax
            assert (probs >= 0).all()

    def test_models_are_compiled_and_trainable(self):
        import numpy as np

        model = build_cnn()
        assert model.optimizer is not None
        history = model.fit(
            np.random.default_rng(0).random((16, 32, 32, 3)),
            np.eye(3)[np.arange(16) % 3],
            epochs=1, verbose=0,
        )
        assert "loss" in history.history


@pytest.fixture(scope="module")
def model():
    """Built once for the whole module — assembling it takes a few seconds."""
    return build_yolov3(input_shape=(416, 416, 3))


@pytest.fixture(scope="module")
def loadable_model():
    """A second instance, because loading weights into a model mutates it."""
    return build_yolov3(input_shape=(416, 416, 3))


class TestYolov3Architecture:
    def test_three_detection_scales(self, model):
        assert len(model.outputs) == 3

    def test_output_grids_and_depth(self, model):
        # 416 downsampled by 32, 16 and 8; 255 = 3 anchors x (4 + 1 + 80).
        assert [tuple(o.shape) for o in model.outputs] == [
            (None, 13, 13, 255),
            (None, 26, 26, 255),
            (None, 52, 52, 255),
        ]

    def test_parameter_count_matches_the_published_network(self, model):
        # YOLOv3 as released has 62,001,757 parameters. Any mismatch means the
        # layer table has drifted from the architecture the weights expect.
        assert model.count_params() == 62_001_757

    def test_seventy_five_convolutions(self, model):
        convs = [layer for layer in model.layers if layer.__class__.__name__ == "Conv2D"]
        assert len(convs) == 75

    def test_only_the_detection_heads_carry_a_bias(self, model):
        biased = sorted(
            int(layer.name.split("_")[1])
            for layer in model.layers
            if layer.__class__.__name__ == "Conv2D" and layer.use_bias
        )
        assert biased == list(DETECTION_LAYERS)

    def test_accepts_any_multiple_of_32(self):
        model = build_yolov3(input_shape=(320, 320, 3))
        assert tuple(model.outputs[0].shape) == (None, 10, 10, 255)


class TestDarknetWeightLoading:
    """The conversion path, exercised without the 237 MB download.

    A real ``yolov3.weights`` is 248,007,048 bytes: a 20-byte header followed by
    exactly 62,001,757 float32 values, one per model parameter. Synthesising a
    file of that shape checks the byte accounting all the way through — if the
    architecture and the reader disagree anywhere, the loader runs out of values
    early or finishes with some left over.
    """

    HEADER_BYTES = 3 * 4 + 8
    PARAMS = 62_001_757

    def _write_weights(self, path, n_floats, major=0, minor=2, revision=0, seen=32013312):
        import numpy as np

        with open(path, "wb") as handle:
            np.array([major, minor, revision], dtype=np.int32).tofile(handle)
            np.array([seen], dtype=np.int64).tofile(handle)
            # Small values keep the synthetic batch-norm variances benign.
            (np.arange(n_floats, dtype=np.float32) % 7 * 1e-3).tofile(handle)
        return path

    def test_synthetic_file_matches_the_official_size(self, tmp_path):
        path = self._write_weights(tmp_path / "yolov3.weights", self.PARAMS)
        assert path.stat().st_size == 248_007_048

    def test_download_script_expects_that_size(self):
        # Guards against the constant in the script drifting from the network.
        import importlib.util
        import pathlib

        script = pathlib.Path(__file__).parent.parent / "scripts" / "download_assets.py"
        spec = importlib.util.spec_from_file_location("download_assets", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module.EXPECTED_BYTES == self.PARAMS * 4 + self.HEADER_BYTES

    def test_loader_consumes_every_value(self, tmp_path, loadable_model):
        from roadvision.darknet import load_darknet_weights

        path = self._write_weights(tmp_path / "yolov3.weights", self.PARAMS)
        # Raises if anything is left unread, so reaching here means the stream
        # and the architecture agree exactly.
        load_darknet_weights(loadable_model, str(path), verbose=False)

    def test_short_file_is_rejected(self, tmp_path, loadable_model):
        from roadvision.darknet import load_darknet_weights

        path = self._write_weights(tmp_path / "short.weights", self.PARAMS - 1000)
        with pytest.raises(ValueError, match="exhausted"):
            load_darknet_weights(loadable_model, str(path), verbose=False)

    def test_long_file_is_rejected(self, tmp_path, loadable_model):
        from roadvision.darknet import load_darknet_weights

        path = self._write_weights(tmp_path / "long.weights", self.PARAMS + 1000)
        with pytest.raises(ValueError, match="left unread"):
            load_darknet_weights(loadable_model, str(path), verbose=False)

    def test_header_version_is_parsed(self, tmp_path):
        from roadvision.darknet import WeightReader

        path = self._write_weights(tmp_path / "v.weights", 100, major=0, minor=2, revision=5)
        assert WeightReader(str(path)).version == (0, 2, 5)
