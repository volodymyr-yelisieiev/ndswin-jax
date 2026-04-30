"""Tests for RandAugment integration and Mixup/CutMix activation."""


def test_auto_augment_config_field():
    """Verify DataConfig accepts the auto_augment field."""
    from ndswin.config import DataConfig

    config = DataConfig(auto_augment="rand-m9-mstd0.5")
    assert config.auto_augment == "rand-m9-mstd0.5"


def test_auto_augment_none_by_default():
    """Verify auto_augment defaults to None."""
    from ndswin.config import DataConfig

    config = DataConfig()
    assert config.auto_augment is None


def test_mixup_alpha_positive():
    """Verify DataConfig accepts positive mixup_alpha."""
    from ndswin.config import DataConfig

    config = DataConfig(mixup_alpha=0.8, cutmix_alpha=1.0)
    assert config.mixup_alpha == 0.8
    assert config.cutmix_alpha == 1.0


def test_mixup_or_cutmix_import():
    """Verify MixupOrCutmix can be imported and instantiated."""
    from ndswin.training.augmentation import MixupOrCutmix

    transform = MixupOrCutmix(
        p=0.5,
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        num_classes=10,
    )
    assert transform is not None
