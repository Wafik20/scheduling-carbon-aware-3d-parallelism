from carbon_aware_3d.compute_power import PowerMonitor


def test_results_average_watts_not_training_steps():
    monitor = PowerMonitor()
    monitor._readings = [("first", 10, 300.0), ("second", 20, 500.0)]

    average_watts, readings = monitor.results()

    assert average_watts == 400.0
    assert readings == monitor._readings


def test_results_without_samples():
    assert PowerMonitor().results() == (0.0, [])
