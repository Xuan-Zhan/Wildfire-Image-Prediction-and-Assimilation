import pytest
from unittest.mock import patch, MagicMock
import torch
from atlaswildfiretool.forecasting.model_train import train_model


@pytest.fixture
def setup_data():
    """
    Fixture to set up mock data and return it.

    Returns
    -------
    tuple
        A tuple containing mock objects for model, train_loader, test_loader, criterion, optimizer, num_epochs, and device.
    """
    model = MagicMock()
    train_loader = MagicMock()
    test_loader = MagicMock()
    criterion = MagicMock()
    optimizer = MagicMock()
    num_epochs = 5
    device = "cpu"
    return model, train_loader, test_loader, criterion, optimizer, num_epochs, device


def test_train_model(setup_data):
    """
    Test to check that train_model function calls the necessary functions the correct number of times.

    Parameters
    ----------
    setup_data : fixture
        The fixture that sets up the mock data.

    This test ensures that:
    - The model is moved to the correct device.
    - The model's train and eval methods are called the correct number of times.
    - The torch.save function is called to save the model.
    """
    model, train_loader, test_loader, criterion, optimizer, num_epochs, device = (
        setup_data
    )

    # Create mock data for the DataLoader
    mock_train_data = [
        (torch.randn(1, 3, 3, requires_grad=True), torch.randn(1, 3, 3))
        for _ in range(5)
    ]
    mock_test_data = [(torch.randn(1, 3, 3), torch.randn(1, 3, 3)) for _ in range(2)]
    train_loader.__len__.return_value = len(mock_train_data)
    test_loader.__len__.return_value = len(mock_test_data)
    train_loader.__iter__.return_value = iter(mock_train_data)
    test_loader.__iter__.return_value = iter(mock_test_data)

    # Mocking the loss computation to return a float value
    def mock_criterion(output, target):
        return torch.tensor(1.0, requires_grad=True)

    criterion.side_effect = mock_criterion

    def mock_forward(x):
        return torch.randn_like(x, requires_grad=True)

    model.side_effect = mock_forward

    with patch.object(model, "train"), patch.object(model, "eval"), patch.object(
        model, "to", return_value=model
    ), patch("torch.save") as mock_save:

        # Run the train_model function
        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            device=device,
        )

        # Check that the model was moved to the correct device
        model.to.assert_called_with(device)

        # Check that the model's train and eval methods were called
        assert model.train.call_count == num_epochs
        assert model.eval.call_count == num_epochs

        # Check that torch.save was called (model saving functionality)
        assert mock_save.called
