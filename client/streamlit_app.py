from src.infer import forecast, load_features_df
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))


st.set_page_config(page_title="Toronto Bikeshare Forecast", layout="wide")
st.title("Toronto Bikeshare — Station Forecast")

df_all = load_features_df()
stations = sorted(df_all["start_station_id"].unique().tolist())
sid = st.selectbox("Station ID", [str(s) for s in stations], index=0)
horizon = st.slider("Forecast horizon (days)", 7, 30, 14, 1)

if st.button("Forecast"):
    sid_int = int(sid)
    fc = forecast(sid_int, horizon)
    hist = (df_all[df_all["start_station_id"] == sid_int]
            .sort_values("trip_date")[["trip_date", "rides"]]
            .tail(30)
            .rename(columns={"rides": "hist_rides"}))
    plot_df = pd.merge(hist, fc, on="trip_date", how="outer").set_index(
        "trip_date").sort_index()
    st.subheader(f"Station {sid}: last 30 days and next {horizon}")
    st.line_chart(plot_df)
    st.subheader("Forecast table")
    st.dataframe(fc)
    csv = fc.to_csv(index=False).encode("utf-8")
    st.download_button("Download forecast CSV", csv,
                       file_name=f"forecast_{sid}.csv", mime="text/csv")
