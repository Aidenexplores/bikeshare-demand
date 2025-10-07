Toronto Bikeshare Demand Forecasting & Station Optimization

Predict daily demand (rides) for each Bike Share Toronto station, explore busiest/least-busy stations, and visualize demand on a Toronto map. The project ships a FastAPI backend and a Streamlit frontend, both containerized and deployable to Google Cloud Run.

App (Streamlit): https://bikeshare-app-628939884230.northamerica-northeast1.run.app

API (FastAPI docs): https://bikeshare-api-628939884230.northamerica-northeast1.run.app/docs



By : Aiden Ebrahimi

Project Overview

Goal: Forecast next-day to multi-day demand per station using historical ridership.
Outputs:

Forecast from any chosen anchor date (including future dates)

Busiest/least-busy station rankings over a selectable window

Heatmap of station activity with station names and coordinates

REST API for programmatic forecasts

Data sources:

Historical ridership CSVs: https://open.toronto.ca/dataset/bike-share-toronto-ridership-data/

Station metadata (names/lat/lon) via GBFS station_information (auto-fetched in app)

Tech stack: Python, pandas, scikit-learn (HistGradientBoostingRegressor), FastAPI, Uvicorn, Streamlit, PyDeck, Docker, Google Cloud Run.
