from modules.data_update.sackmann import sync_sackmann_atp, sync_sackmann_wta
from modules.data_update.tennis_data_odds import download_tennis_data_odds

__all__ = ["sync_sackmann_atp", "sync_sackmann_wta", "download_tennis_data_odds", "build_upcoming"]


def build_upcoming(*args, **kwargs):
    from modules.data_update.upcoming import build_upcoming as _build
    return _build(*args, **kwargs)
