import torch

from solar.ridge import RidgeStatistics, tune_ridge


def test_streaming_ridge_recovers_multioutput_weight_and_intercept():
    generator = torch.Generator().manual_seed(17)
    training_features = torch.randn((80, 3), generator=generator)
    validation_features = torch.randn((30, 3), generator=generator)
    expected_weight = torch.tensor([[1.5, -0.3, 0.8], [-0.5, 0.2, 1.1]])
    expected_bias = torch.tensor([0.7, -1.2])
    training_target = training_features @ expected_weight.T + expected_bias
    validation_target = validation_features @ expected_weight.T + expected_bias

    training = RidgeStatistics.empty(feature_size=3, output_size=2)
    training.update(training_features[:35], training_target[:35])
    training.update(training_features[35:], training_target[35:])
    validation = RidgeStatistics.empty(feature_size=3, output_size=2)
    validation.update(validation_features, validation_target)

    alpha, validation_mse, weight, bias = tune_ridge(
        training, validation, (0.0, 1e-3)
    )
    assert alpha == 0.0
    assert validation_mse < 1e-12
    assert torch.allclose(weight, expected_weight, atol=1e-5, rtol=1e-5)
    assert torch.allclose(bias, expected_bias, atol=1e-5, rtol=1e-5)
