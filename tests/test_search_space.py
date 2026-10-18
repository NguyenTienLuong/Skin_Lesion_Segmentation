from src.hpo.search_space import configuration_from_params


def test_configuration_from_params_adam_bce():
    cfg = configuration_from_params({"batch_size": 8, "optimizer": "Adam", "loss_type": "BCE"})
    assert cfg["batch_size"] == 8
    assert cfg["optimizer"]["name"] == "Adam"
    assert cfg["optimizer"]["weight_decay"] == 0.0
    assert cfg["loss"]["name"] == "BCE"
