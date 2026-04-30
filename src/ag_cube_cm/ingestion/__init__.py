"""ag_cube_cm.ingestion — Data download layer (download-only, no cube-building)."""
from .weather import AgEra5Downloader, CHIRPSDownloader, WeatherDownloadOrchestrator
from .soil import SoilGridsDownloader
from .boundaries import get_admin_boundary, list_admin_units

__all__ = [
    "AgEra5Downloader",
    "CHIRPSDownloader",
    "WeatherDownloadOrchestrator",
    "SoilGridsDownloader",
    "get_admin_boundary",
    "list_admin_units",
]
