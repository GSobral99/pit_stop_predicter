import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import _default_model_path


class TestDefaultModelPath:
    def test_single_circuit_uses_circuit_slug(self):
        path = _default_model_path(['Monza'])
        assert path == 'models/tyre_degradation_monza.joblib'

    def test_single_circuit_with_spaces(self):
        path = _default_model_path(['Belgian Grand Prix'])
        assert path == 'models/tyre_degradation_belgian_grand_prix.joblib'

    def test_multiple_circuits_are_flagged_as_multi(self):
        path = _default_model_path(['Monza', 'Belgian Grand Prix'])
        assert 'multi' in path
        assert 'monza' in path
        assert 'belgian_grand_prix' in path