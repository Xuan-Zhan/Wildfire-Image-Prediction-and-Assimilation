"""test module for generative/training.py."""

import pytest
from unittest.mock import patch, MagicMock, call

from atlaswildfiretool.generative.training import train_vae


@pytest.fixture
def setup_model():
    model = MagicMock()
    optimizer = MagicMock()
    scheduler = MagicMock()
    train_loader = MagicMock()
    val_loader = MagicMock()
    return model, optimizer, scheduler, train_loader, val_loader


def test_train_vae_calls_train_and_validate(setup_model):
    model, optimizer, scheduler, train_loader, val_loader = setup_model
    n_epochs = 5
    device = "cpu"

    with patch("atlaswildfiretool.generative.training.train") as mock_train, patch(
        "atlaswildfiretool.generative.training.validate"
    ) as mock_validate, patch(
        "atlaswildfiretool.generative.training.save_checkpoint"
    ) as mock_save:
        train_vae(
            n_epochs=n_epochs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            model_save_path=None,
            use_liveloss=False,
            device=device,
        )

        # check func calls.
        assert mock_train.call_count == n_epochs
        assert mock_validate.call_count == n_epochs
        assert mock_save.call_count == 0
