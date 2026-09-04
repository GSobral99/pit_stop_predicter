from unittest.mock import MagicMock, patch

import pytest
from src.data_loader import load_race


def _fake_session(event_name, location):
    session = MagicMock()
    session.event = {'EventName': event_name, 'Location': location}
    session.load.return_value = None
    return session


class TestLoadRaceVerification:
    @patch('src.data_loader.fastf1')
    def test_passes_when_gp_matches_event(self, mock_fastf1):
        mock_fastf1.get_session.return_value = _fake_session(
            'Belgian Grand Prix', 'Spa-Francorchamps'
        )
        session = load_race(2023, 'Spa', 'R', verify_contains='Spa')
        assert session is not None

    @patch('src.data_loader.fastf1')
    def test_raises_when_gp_resolves_to_wrong_race(self, mock_fastf1):
        mock_fastf1.get_session.return_value = _fake_session(
            'Spanish Grand Prix', 'Barcelona'
        )
        with pytest.raises(ValueError):
            load_race(2023, 'Spa', 'R', verify_contains='Francorchamps')

    @patch('src.data_loader.fastf1')
    def test_no_verification_by_default(self, mock_fastf1):
        mock_fastf1.get_session.return_value = _fake_session(
            'Spanish Grand Prix', 'Barcelona'
        )
        session = load_race(2023, 'Spa', 'R')
        assert session is not None