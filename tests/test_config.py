from ndswin.config import ExperimentConfig


def test_experiment_config_description():
    """Test that ExperimentConfig correctly parses the description field."""
    # Without description
    config_dict = {"name": "test_exp"}
    config = ExperimentConfig.from_dict(config_dict)
    assert config.name == "test_exp"
    assert config.description == "No description provided"

    # With description
    config_dict = {"name": "test_exp2", "description": "A very important experiment"}
    config = ExperimentConfig.from_dict(config_dict)
    assert config.name == "test_exp2"
    assert config.description == "A very important experiment"

    # Ensure it serializes correctly
    out_dict = config.to_dict()
    assert out_dict["description"] == "A very important experiment"


def test_data_config_image_size():
    from ndswin.config import DataConfig

    # Test that a list passed as image_size becomes a tuple
    config = DataConfig(image_size=[32, 32])
    assert isinstance(config.image_size, tuple)
    assert config.image_size == (32, 32)
