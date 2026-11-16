"""
Biobeat - FastAPI Backend Proxy for Meteomatics API
Securely handles Basic Authentication server-side and aggregates spatial climate data.

Run:
    python -m uvicorn main:app --reload
API Contract:
    GET /api/config
    GET /api/series?track=&hemisphere=&lat=&lon=&hours_back=&hours_fwd=
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.staticfiles import StaticFiles

# Configure professional structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("biobeat.api")

BASE_URL = "https://api.meteomatics.com"
UNIT_MAPPINGS = {
    "C": "°C",
    "mm": "mm",
    "ms": "m/s",
    "cm": "cm",
    "idx": "index",
    "hPa": "hPa",
    "p": "%",
}

# Climate phenomena with high-variability parameters
TRACKS: Dict[str, Dict[str, Any]] = {
    "temp": {
        "label": "Temperature (2m)",
        "params": ["t_2m:C", "t_apparent:C"],
    },
    "humid": {
        "label": "Relative Humidity",
        "params": ["relative_humidity_2m:p"],
    },
    "wind": {
        "label": "Wind Speed (10m)",
        "params": ["wind_speed_10m:ms", "wind_gusts_10m_1h:ms"],
    },
    "press": {
        "label": "Atmospheric Pressure",
        "params": ["msl_pressure:hPa"],
    },
    "rain": {
        "label": "Precipitation",
        "params": ["precip_1h:mm"],
    },
    "sea": {
        "label": "Ocean Current",
        "params": ["ocean_current_speed:ms"],
        "ocean": True,
    },
}

# Hemispheric bounding boxes and stratified land/sea sampling coordinates
# Hemispheric bounding boxes spanning the complete hemisphere (Equator to Pole)
HEMISPHERES: Dict[str, Dict[str, Any]] = {
    "north": {
        "label": "Northern Hemisphere",
        "bounds": [[0, -180], [90, 180]],
        "land": [[45.0, -100.0], [50.0, 10.0], [35.0, 105.0], [55.0, 37.0], [22.0, 78.0], [40.0, -5.0]],
        "sea": [[30.0, -45.0], [35.0, -160.0], [20.0, 65.0], [45.0, -25.0], [15.0, 140.0], [50.0, -170.0]],
    },
    "south": {
        "label": "Southern Hemisphere",
        "bounds": [[-90, -180], [0, 180]],
        "land": [[-15.0, -60.0], [-25.0, 135.0], [-30.0, 25.0], [-8.0, -55.0], [-20.0, 47.0], [-34.0, 19.0]],
        "sea": [[-25.0, 80.0], [-30.0, -40.0], [-20.0, -110.0], [-35.0, 150.0], [-15.0, -10.0], [-45.0, 60.0]],
    },
}

app = FastAPI(
    title="Biobeat API",
    description="Earth Information Jukebox - Spatial Climate Telemetry Proxy",
    version="2.0.0",
)


@app.get("/api/config", summary="Retrieve configuration metadata")
def get_config() -> Dict[str, Any]:
    """Provide available tracking metrics and hemispheric regions to the frontend."""
    return {"hemispheres": HEMISPHERES, "tracks": TRACKS}


async def _fetch_point_data(
    client: httpx.AsyncClient, start: str, end: str, params: List[str], lat: float, lon: float
) -> Tuple[str, List[str], List[float]]:
    """Iterate through fallback parameters for a specific coordinate point until a valid response is received."""
    last_response = None
    for param in params:
        url = f"{BASE_URL}/{start}--{end}:PT1H/{param}/{lat},{lon}/json"
        try:
            response = await client.get(url)
            if response.status_code == status.HTTP_200_OK:
                payload = response.json()
                dates = [row["date"] for row in payload["data"][0]["coordinates"][0]["dates"]]
                values = [row["value"] for row in payload["data"][0]["coordinates"][0]["dates"]]
                return param, dates, values
            last_response = response
        except httpx.RequestError as exc:
            logger.warning(f"Network error querying parameter {param} at ({lat}, {lon}): {exc}")

    status_code = last_response.status_code if last_response else status.HTTP_502_BAD_GATEWAY
    error_text = last_response.text[:160] if last_response else "No response from upstream provider"
    logger.error(f"Failed to fetch data for params {params} at ({lat}, {lon}): {error_text}")
    raise HTTPException(status_code=status_code, detail=f"Meteomatics upstream error: {error_text}")


@app.get("/api/series", summary="Retrieve aggregated time-series telemetry")
async def get_series(
    track: str = Query(..., description="Target track identifier (e.g., temp, wind)"),
    hemisphere: Optional[str] = Query(None, description="Hemisphere key (north or south)"),
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Custom latitude coordinate"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Custom longitude coordinate"),
    hours_back: int = Query(72, ge=1, le=168, description="Historical observation window in hours"),
    hours_fwd: int = Query(24, ge=1, le=168, description="Forecast window in hours"),
) -> Dict[str, Any]:
    """Fetch spatial time series data across multiple sample points and compute aggregate averages."""
    if track not in TRACKS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid track parameter.")

    meteomatics_user = os.getenv("METEOMATICS_USER")
    meteomatics_pass = os.getenv("METEOMATICS_PASS")
    if not meteomatics_user or not meteomatics_pass:
        logger.critical("Meteomatics credentials missing from server environment variables.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: Missing API credentials.",
        )

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    time_format = "%Y-%m-%dT%H:%M:%SZ"
    start_time = (now - timedelta(hours=hours_back)).strftime(time_format)
    end_time = (now + timedelta(hours=hours_fwd)).strftime(time_format)

    # Determine sampling points based on region or coordinates
    if hemisphere in HEMISPHERES:
        hemi_config = HEMISPHERES[hemisphere]
        is_ocean = TRACKS[track].get("ocean", False)
        points = hemi_config["sea"] if is_ocean else hemi_config["land"]
        region_label = f"{hemi_config['label']} ({'Ocean' if is_ocean else 'Surface'})"
        bounds = hemi_config["bounds"]
    elif lat is not None and lon is not None:
        points = [[lat, lon]]
        region_label = f"Custom Point ({lat}, {lon})"
        bounds = None
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid hemisphere ('north' or 'south') or valid geographic coordinates (lat, lon) must be provided.",
        )

    track_params = TRACKS[track]["params"]
    
    # Asynchronously fetch data across all sample points concurrently
    async with httpx.AsyncClient(
        auth=(meteomatics_user, meteomatics_pass),
        timeout=30.0
    ) as client:
        fetch_tasks = [_fetch_point_data(client, start_time, end_time, track_params, pt[0], pt[1]) for pt in points]
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    # Filter out failed point queries
    successful_results = [res for res in results if not isinstance(res, Exception)]
    if not successful_results:
        first_error = results[0]
        if isinstance(first_error, HTTPException):
            raise first_error
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(first_error))

    used_param = successful_results[0][0]
    dates = successful_results[0][1]
    
    # Compute spatial mean across valid sample points for each timestamp
    aggregated_values: List[Optional[float]] = []
    for i in range(len(dates)):
        timestamp_values = [
            res[2][i] for res in successful_results if i < len(res[2]) and res[2][i] is not None
        ]
        if timestamp_values:
            aggregated_values.append(round(sum(timestamp_values) / len(timestamp_values), 3))
        else:
            aggregated_values.append(None)

    active_points = [points[i] for i, res in enumerate(results) if not isinstance(res, Exception)]
    unit_key = used_param.split(":")[-1]

    return {
        "param": used_param,
        "unit": UNIT_MAPPINGS.get(unit_key, ""),
        "dates": dates,
        "values": aggregated_values,
        "region": {
            "hemisphere": hemisphere,
            "label": region_label,
            "bounds": bounds,
            "points": active_points,
            "samples": len(active_points),
        },
    }


# Mount static frontend assets if directory exists
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")