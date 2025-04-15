import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

def get_weather_data():
    # Setup client
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # API params
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 24.4512,
        "longitude": 54.397,
        "hourly": [ #CONTAINS 6 FEATURES
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation_probability",
            "cloud_cover_high",
            "wind_direction_80m"
        ],
        "timezone": "auto",
        "past_days": 30,
        "wind_speed_unit": "ms"
    }

    # Request
    response = openmeteo.weather_api(url, params=params)[0]
    hourly = response.Hourly()

    # Extract variables
    data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "apparent_temperature": hourly.Variables(2).ValuesAsNumpy(),
        "precipitation_probability": hourly.Variables(3).ValuesAsNumpy(),
        "cloud_cover_high": hourly.Variables(4).ValuesAsNumpy(),
        "wind_direction_80m": hourly.Variables(5).ValuesAsNumpy()
    }

    df = pd.DataFrame(data)
    return df