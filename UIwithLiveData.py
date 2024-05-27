import folium
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import dask.dataframe as dd
import dask.array as da
import geopandas as gpd
import matplotlib.pyplot as plt
from folium.plugins import HeatMap
import altair as alt
from shapely import wkt
from shapely.geometry import Polygon, Point
import numpy as np
from joblib import load
from datetime import datetime, timedelta
import pyproj
from shapely.geometry import Point, Polygon
import geopandas as gpd
from shapely.ops import transform
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from sodapy import Socrata
import math
import requests
# Set Streamlit page configuration
st.set_page_config(layout="wide")
# Month,Day,hour,SPEED,school_count,park_count,MeanTemp,MinTemp,MaxTemp,DewPoint,Percipitation,WindSpeed,MaxSustainedWind,Rain,SnowDepth,SnowIce,NUMBER OF PERSONS INJURED,NUMBER OF PERSONS KILLED,Accident Severity,numIntersections
severity = {0: 'Extreme Hot Spot', 1: 'Currently Safe', 1: 'Potential Hot Spot', 3: 'Hot Spot'}

zipDict = {10001: 0,10002: 1,10003: 2,10004: 3,10007: 4,10009: 5,10010: 6,10011: 7,10013: 8,10014: 9,10016: 10,10019: 11,10021: 12,10022: 13,10023: 14,10024: 15,10025: 16,10026: 17,10027: 18,10028: 19,10029: 20,10030: 21,10031: 22,10032: 23,10033: 24,10034: 25,10035: 26,10036: 27,10037: 28,10038: 29,10039: 30,10040: 31,10065: 32,10128: 33,10301: 34,10302: 35,10303: 36,10304: 37,10305: 38,10306: 39,10307: 40,10308: 41,10309: 42,10310: 43,10312: 44,10314: 45,10451: 46,10452: 47,10453: 48,10454: 49,10455: 50,10456: 51,10457: 52,10458: 53,10459: 54,10460: 55,10461: 56,10462: 57,10463: 58,10464: 59,10465: 60,10466: 61,10467: 62,10468: 63,10469: 64,10470: 65,10471: 66,10472: 67,10473: 68,10474: 69,10475: 70,11004: 71,11101: 72,11102: 73,11103: 74,11104: 75,11105: 76,11106: 77,11201: 78,11203: 79,11204: 80,11205: 81,11206: 82,11207: 83,11208: 84,11209: 85,11210: 86,11211: 87,11212: 88,11213: 89,11214: 90,11215: 91,11216: 92,11217: 93,11218: 94,11219: 95,11220: 96,11221: 97,11222: 98,11223: 99,11224: 100,11225: 101,11226: 102,11228: 103,11229: 104,11230: 105,11231: 106,11232: 107,11233: 108,11234: 109,11235: 110,11236: 111,11237: 112,11238: 113,11239: 114,11354: 115,11355: 116,11356: 117,11357: 118,11358: 119,11360: 120,11361: 121,11362: 122,11363: 123,11364: 124,11365: 125,11366: 126,11367: 127,11368: 128,11369: 129,11370: 130,11372: 131,11373: 132,11374: 133,11375: 134,11377: 135,11378: 136,11379: 137,11385: 138,11411: 139,11412: 140,11413: 141,11414: 142,11415: 143,11416: 144,11417: 145,11418: 146,11419: 147,11420: 148,11421: 149,11422: 150,11423: 151,11426: 152,11427: 153,11428: 154,11429: 155,11432: 156,11433: 157,11434: 158,11435: 159,11436: 160,11691: 161,11692: 162,11693: 163,11694: 164}

boroughDict = {'BRONX':0,'BROOKLYN':1,'MANHATTAN':2,'QUEENS':3,'STATEN ISLAND':4}
includeRain = includeTraffic = False

def includeRain():
    includeRain = True

def getClient(source,token):
    client = Socrata(source,
                     token,
                     timeout = 100)
    return client

def fetchData(borough, currentDateTime, data = "i4gi-tjb9",timeDelta = 48):
    limit = 5000
    offset = 0

    time2 = currentDateTime.isoformat()
    time1 = currentDateTime + timedelta(hours=-timeDelta)
    time1 = time1.isoformat()
    client = getClient("data.cityofnewyork.us","<NYC API TOKEN>")

    results = client.get(data,
                         borough = borough,
                         limit = limit,
                         offset = offset,
                         where = f"DATA_AS_OF between '{time1}' and '{time2}'"
                         )

    pdDf = pd.DataFrame.from_records(results)
    results_df = dd.from_pandas(pdDf, npartitions = 5)
    return results_df

def fetchHMData(currentDateTime, data = "i4gi-tjb9",timeDelta = 48):
    limit = 10000
    offset = 0

    time2 = currentDateTime.isoformat()
    time1 = currentDateTime + timedelta(hours=-timeDelta)
    time1 = time1.isoformat()
    client = getClient("data.cityofnewyork.us","<NYC API TOKEN>")

    results = client.get(data,
                         limit = limit,
                         offset = offset,
                         where = f"DATA_AS_OF between '{time1}' and '{time2}'")

    pdDf = pd.DataFrame.from_records(results)
    results_df = dd.from_pandas(pdDf, npartitions = 5)
    return results_df

def cleanData(data):
    columns = [
        "id",
        "link_id",
        "encoded_poly_line_lvls",
        "owner",
        "transcom_id",
        "borough",
        "link_name"
    ]
    data = data.drop(columns = columns)

    return data

def getExplodedData(trafficDf):
    trafficDf["link_points"] = trafficDf['link_points'].apply(lambda x: x.split(" ")[:-1], meta = list)
    trafficDf = trafficDf.explode('link_points')
    trafficDf['pointWKT'] = trafficDf['link_points'].apply(lambda x: f'Point({x.split(",")[1]} {x.split(",")[0]})' if len(x.split(",")) == 2 else np.nan)
    trafficDf = trafficDf.dropna(subset=['pointWKT'])
    return trafficDf

def getLiveTrafficLatLng(trafficDf):
    trafficDf['speed'] = trafficDf['speed'].astype('float64')
    trafficDf = trafficDf[trafficDf['speed'] < 15]
    trafficDf["link_points"] = trafficDf['link_points'].apply(lambda x: x.split(" ")[:-1], meta = list)
    trafficDf = trafficDf.explode('link_points')
    trafficDf['lat'] = trafficDf['link_points'].apply(lambda x: float(x.split(",")[0]) if len(x.split(",")) == 2 else np.nan)
    trafficDf['lng'] = trafficDf['link_points'].apply(lambda x: float(x.split(",")[1]) if len(x.split(",")) == 2 else np.nan)
    # trafficDf['pointWKT'] = trafficDf['link_points'].apply(lambda x: f'Point({x.split(",")[1]} {x.split(",")[0]})' if len(x.split(",")) == 2 else np.nan)
    trafficDf = trafficDf.dropna(subset=['lat','lng'])
    return trafficDf

def getInsidePolyPoints(trafficDf, polygon):
    trafficDf['insidePoly'] = trafficDf['pointWKT'].apply(lambda x: polygon.contains(wkt.loads(x)),meta=bool)
    trafficDf = trafficDf[trafficDf['insidePoly'] == True]
    return trafficDf

def getAvgSpeed(trafficDf):
    # dd.to_numeric(trafficDf['speed'], errors='coerce')
    speeds = trafficDf['speed'].values
    computed_speeds = speeds.compute()
    computed_speeds = computed_speeds.astype(float)
    return da.mean(computed_speeds).compute()

def getAvgTrafficInZipcode(borough, polyWKT):
    trafficDf = fetchData(borough, datetime.now())
    if len(trafficDf) != 0:
        trafficDf = cleanData(trafficDf)
        trafficDf = getExplodedData(trafficDf)

        poly = wkt.loads(polyWKT)
        trafficDf = getInsidePolyPoints(trafficDf, poly)
        avgSpeed = getAvgSpeed(trafficDf)
        return avgSpeed
    return -1

def liveTrafficSpeed(borough, polyWKT):
    return getAvgTrafficInZipcode(borough, polyWKT)

def getMapObject():
    m = folium.Map(location=[40.730610, -73.935242], zoom_start=10)
    return m

def getBounds(st_data):
    if 'bounds' in st_data:
        try:
            sw = (st_data['bounds']['_southWest']['lng'], st_data['bounds']['_southWest']['lat'])
            ne = (st_data['bounds']['_northEast']['lng'], st_data['bounds']['_northEast']['lat'])
            return Polygon([(sw[0], sw[1]), (sw[0], ne[1]), (ne[0], ne[1]), (ne[0], sw[1])])
        except TypeError:
            st.error("Error in retrieving map bounds. Please adjust the map view.")
            return None
    return None

def getCenter(st_data):
    if 'center' in st_data:
        return (st_data['center']['lat'],st_data['center']['lng'])
    return None

def loadAllData():
    networkZip = pd.read_csv("data/networks.csv")
    combinedData = pd.read_csv('data/finalCombinedData.csv')
    return networkZip, combinedData

def getIntersections(networkZip, polyWKT):
    try:
        networkZip['geometry'] = networkZip['geometry'].apply(wkt.loads)
    except:
        pass
    # zipcodeData['the_geom'] = zipcodeData['the_geom'].apply(wkt.loads)
    if isinstance(polyWKT, str):
        polyWKT = wkt.loads(polyWKT)
    polyDf = pd.DataFrame({'polygon_geom':[polyWKT]})
    gdf_points = gpd.GeoDataFrame(networkZip, geometry='geometry')
    gdf_polygons = gpd.GeoDataFrame(polyDf, geometry='polygon_geom')

    # Set a common Coordinate Reference System
    gdf_points.crs = gdf_polygons.crs = "EPSG:4326"
    result = gpd.sjoin(gdf_points, gdf_polygons, how="inner", op='within')
    return len(result)

def getDaskArray():
    pass

def create_circle(latitude, longitude, radius_meters):
        proj = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=latitude, lon_0=longitude)
        project_to_aeqd = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'), proj)
        project_to_lonlat = partial(pyproj.transform, proj, pyproj.Proj(init='epsg:4326'))
        center_point = transform(project_to_aeqd, Point(longitude, latitude))
        circle = center_point.buffer(radius_meters)
        circle_lonlat = transform(project_to_lonlat, circle)

        return circle_lonlat

weather_cache = {}
def fetch_weather_data(zipcode):
    current_time = datetime.now()
    if zipcode in weather_cache and (current_time - weather_cache[zipcode]['time']).seconds < 900:
        print("Using cached data")
        return weather_cache[zipcode]['data']

    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{zipcode}/2024-05-08"
    params = {"key": "<Weather API TOKEN>"}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        weather_cache[zipcode] = {'time': current_time, 'data': response.json()}
        return response.json()
    else:
        response.raise_for_status()

def getArea(center):
    circle = create_circle(center[0], center[1], 300)
    centerData = gdf[gdf.geometry.within(circle)]
    return centerData, centerData.iloc[0]['zipcode'], centerData.iloc[0]['BOROUGH']

def getWeatherVec(zipcode, withinBounds, rain):
    try:
        weather = fetch_weather_data(zipcode)
    except:
        MeanTemp = withinBounds['MeanTemp'].mean()
        MinTemp = withinBounds['MinTemp'].mean()
        MaxTemp = withinBounds['MaxTemp'].mean()
        DewPoint = withinBounds['DewPoint'].mean()
        Percipitation = withinBounds['Percipitation'].mean()
        WindSpeed = withinBounds['WindSpeed'].mean()
        MaxSustainedWind = withinBounds['MaxSustainedWind'].mean()
        Rain = withinBounds['Rain'].mean()
        SnowDepth = withinBounds['SnowDepth'].mean()
        SnowIce = withinBounds['SnowIce'].mean()
        school_count = math.ceil(withinBounds['school_count'].mean())
        park_count = math.ceil(withinBounds['park_count'].mean())
        return [school_count,park_count,MeanTemp,MinTemp,MaxTemp,DewPoint,Percipitation,WindSpeed,MaxSustainedWind,Rain,SnowDepth,SnowIce], ('Not Available', int(MeanTemp))

    if 'currentConditions' in weather:
        weatherData = weather['currentConditions']
    else:
        weatherData = weather['days'][0]['hours'][-1]

    school_count = math.ceil(withinBounds['school_count'].mean())
    park_count = math.ceil(withinBounds['park_count'].mean())
    MeanTemp = weatherData['feelslike']
    MinTemp = weatherData['temp']
    MaxTemp = weatherData['temp']
    DewPoint = weatherData['dew']
    Percipitation = weatherData['precipprob']
    WindSpeed = weatherData['windspeed']
    MaxSustainedWind = weatherData['windgust']
    Rain = weatherData['precip'] if weatherData['precip'] != 0 else 1 if rain else 0
    SnowDepth = weatherData['snowdepth']
    SnowIce = weatherData['snow']
    conditions = weatherData['conditions']

    return [school_count,park_count,MeanTemp,MinTemp,MaxTemp,DewPoint,Percipitation,WindSpeed,MaxSustainedWind,Rain,SnowDepth,SnowIce], (conditions, MeanTemp)

def getDatevec():
    month = datetime.now().month
    day = datetime.now().day
    hour = datetime.now().hour
    return [month,day,hour]

def getSpeedVec(borough, bounds, zipcode, hour, traffic):
    speeds = liveTrafficSpeed(borough, bounds)
    if speeds == -1:
        if traffic:
            speeds = 5
        else:
            speeds = gdf[(gdf['zipcode'] == zipcode) & (gdf['hour'] == hour)]['SPEED'].mean()
    return speeds

def getZandBVec(zipcode, borough):
    boroughVec = [0] * 5
    boroughVec[boroughDict[borough]] = 1

    zipcodeVec = [0] * len(zipDict)
    zipcodeVec[zipDict[zipcode]] = 1

    return zipcodeVec, boroughVec

def getSeverity(gdf,center,bounds, traffic, rain):
    # knn_model = load('models/newknn_model.joblib')
    rf_model = load('models/newrf_model.joblib')

    inputArray = []

    withinBounds, zipcode, borough = getArea(center)
    dateVec = getDatevec()
    speeds = getSpeedVec(borough, bounds, zipcode, dateVec[-1], traffic)
    weatherVec, displayWeather = getWeatherVec(zipcode, withinBounds, rain)
    intersections = getIntersections(networkZip,bounds)
    zipcodeVec, boroughVec = getZandBVec(zipcode, borough)

    inputArray.extend(dateVec)

    inputArray.append(speeds)

    inputArray.extend(weatherVec)

    inputArray.append(intersections)

    inputArray.extend(boroughVec)

    inputArray.extend(zipcodeVec)

    darray = da.asarray([inputArray])
    result = rf_model.predict(darray)

    return severity[result[0]], intersections, int(speeds), displayWeather
    # currentVector = getDaskArray()


@st.cache_data()
def crashHeatmap(data, _base_map):
    heatmap_layer = HeatMap(data)
    feature_group = folium.FeatureGroup(name='Crash Layer', show=False)
    feature_group.add_child(heatmap_layer).add_to(_base_map)
      # Add layer control to toggle on/off
    return _base_map

@st.cache_data()#ttl = "1m30s")
def trafficHeatmap(_base_map):
    data = fetchHMData(datetime.now())
    trafficHeatmapData = []
    if len(data) == 0:
        trafficHeatmapData = []
        # return _base_map
    else:
        trafficData = cleanData(data)
        trafficData = getLiveTrafficLatLng(trafficData)
        array = trafficData[['lat','lng']].values.compute()
        trafficHeatmapData = array
    trafficLayer = HeatMap(trafficHeatmapData)
    tl = folium.FeatureGroup(name='Traffic Layer', show=False)
    tl.add_child(trafficLayer).add_to(_base_map)
    return _base_map


# Main display section
networkZip, combinedData = loadAllData()
combinedData['datetime'] = pd.to_datetime(combinedData[['Year', 'Month', 'Day', 'hour']])
gdf = gpd.GeoDataFrame(
    combinedData,
    geometry=gpd.points_from_xy(combinedData['LONGITUDE'], combinedData['LATITUDE'])
)

with st.container():
    st.header("Crash No Mo!", divider='rainbow')
    col1, col2 = st.columns(2)  # Specify the width ratios

    with col1:
        m = getMapObject()
        heatmap_data = combinedData[['LATITUDE', 'LONGITUDE']].values.tolist()
        m = crashHeatmap(heatmap_data, m)
        # with ThreadPoolExecutor() as executor:
        m = trafficHeatmap(m)
        folium.LayerControl().add_to(m)
            # future = executor.submit(trafficHeatmap)
        st_data = st_folium(m, width = '50%', height=500)  # Adjust width and height accordingly

    bounds_polygon = getBounds(st_data)
    center = getCenter(st_data)

    with col2:

        includeRain = st.toggle("With Heavy Rain")
        includeTraffic = st.toggle("With High Traffic")

        severity, intersections, speed, weather = getSeverity(gdf, center,bounds_polygon, includeTraffic, includeRain)
        row1_col1, row1_col2 = st.columns(2, gap="small")

        with row1_col1:
            st.markdown(f"""
            <div style="margin: 20px; padding: 10px; height: 100px;">
                <h6>Crash Severity Prediction</h6>
                <h4>{severity}</h4>
            </div>
            """, unsafe_allow_html=True)
        with row1_col2:
            st.markdown(f"""
            <div style="margin: 20px; padding: 10px; height: 100px;">
                <h6>Avg Traffic Speeds</h6>
                <h4>{speed} MPH</h4>
            </div>
            """, unsafe_allow_html=True)

        row2_col1, row2_col2 = st.columns(2, gap="small")
        with row2_col1:
            st.markdown(f"""
            <div style="margin: 20px; padding: 10px; height: 100px;">
                <h6>Weather</h6>
                <h4>{weather[1]} °F, {weather[0]}</h4>
            </div>
            """, unsafe_allow_html=True)
        with row2_col2:
            st.markdown(f"""
            <div style="margin: 20px; padding: 10px; height: 100px;">
                <h6>Intersections</h6>
                <h4>{intersections}</h4>
            </div>
            """, unsafe_allow_html=True)

if st_data:
    bounds_polygon = getBounds(st_data)
    print("bounds_polygon")
    print(bounds_polygon)
    if bounds_polygon:
        filtered_data = gdf[gdf.geometry.within(bounds_polygon)]
        filtered_data['Rain Category'] = pd.cut(combinedData['Rain'], bins=[-np.inf, 0.1, 0.9, np.inf], labels=['No Rain', 'Drizzle', 'Heavy Rain'])


        if not filtered_data.empty:
            col1, col2 = st.columns(2)

            with col1:
                severity_rain = pd.crosstab(filtered_data['Accident Severity'], filtered_data['Rain Category'])
                severity_rain = severity_rain.reset_index()

                # Rename the columns to ensure they are treated as nominal data types
                severity_rain.columns = ['Accident Severity'] + severity_rain.columns[1:].tolist()

                # Custom color scale for rain categories
                rain_colors = {
                    "No Rain": "#add8e6",
                    "Drizzle": "#0000ff",
                    "Heavy Rain": "#d62728"
                }

                # Altair chart
                chart = alt.Chart(severity_rain.melt('Accident Severity', var_name='Rain Category', value_name='Count')).mark_bar().encode(
                    x='Accident Severity:N',
                    y=alt.Y('Count:Q', title='Number of Accidents'),
                    color=alt.Color('Rain Category:N', scale=alt.Scale(domain=list(rain_colors.keys()), range=list(rain_colors.values())), legend=alt.Legend(orient='bottom')),
                    tooltip=['Accident Severity', 'Rain Category', 'Count']
                ).interactive()

                # Display the chart in Streamlit
                st.subheader("Rain vs. Accident Severity")
                st.altair_chart(chart, use_container_width=True)

                severity_hour = pd.crosstab(combinedData['datetime'].dt.hour, filtered_data['Accident Severity']).reset_index()

                expected_severity_levels = ['Low Severity', 'Medium Severity', 'High Severity', 'Extreme']
                actual_severity_levels = severity_hour.columns[1:]  # Excluding 'index' or 'Hour' if you reset_index

                # Rename columns
                new_column_names = ['Hour'] + expected_severity_levels[:len(actual_severity_levels)]
                severity_hour.columns = new_column_names

                # Prepare custom color scale
                custom_color_scale = alt.Scale(domain=expected_severity_levels, range=['#add8e6', '#0000ff', '#ffcccc', '#d62728'])

                # Create Altair chart
                chart = alt.Chart(severity_hour).mark_bar().encode(
                    x=alt.X('Hour:O', title='Hour of the Day'),
                    y=alt.Y('Count:Q', title='Accident Count'),
                    color=alt.Color('Severity:N', scale=custom_color_scale,legend=alt.Legend( orient='bottom'))
                ).transform_fold(
                    fold=expected_severity_levels,
                    as_=['Severity', 'Count']
                ).interactive()

                # Display the chart in Streamlit
                st.subheader("Accident Severity vs. Hour of Day")
                st.altair_chart(chart, use_container_width=True)

                st.subheader("Number of Persons Injured Distribution")
                injured_counts = filtered_data['NUMBER OF PERSONS INJURED'].value_counts()
                st.bar_chart(injured_counts)

            with col2:
                # Accident Severity vs Month

                total_intersections = filtered_data['numIntersections'].sum()
                total_schools = filtered_data['school_count'].sum()
                total_parks = filtered_data['park_count'].sum()

                combinedData['Month'] = combinedData['datetime'].dt.month  # Extract the month from the datetime
                severity_month = pd.crosstab(combinedData['Month'], filtered_data['Accident Severity']).reset_index()

                # Assuming that all expected severity levels are the same as before
                expected_severity_levels = ['Low Severity', 'Medium Severity', 'High Severity', 'Extreme']
                actual_severity_levels = severity_month.columns[1:]  # Columns after 'Month'

                # Rename columns to match expected levels for consistency
                new_column_names = ['Month'] + expected_severity_levels[:len(actual_severity_levels)]
                severity_month.columns = new_column_names

                # Prepare custom color scale
                custom_color_scale = alt.Scale(domain=expected_severity_levels, range=['#add8e6', '#0000ff', '#ffcccc', '#d62728'])

                # Create an Altair chart for months
                month_chart = alt.Chart(severity_month).mark_bar().encode(
                    x=alt.X('Month:O', title='Month of the Year'),
                    y=alt.Y('Count:Q', title='Accident Count'),
                    color=alt.Color('Severity:N', scale=custom_color_scale, legend=alt.Legend(orient='bottom'))
                ).transform_fold(
                    fold=expected_severity_levels,
                    as_=['Severity', 'Count']
                ).interactive()
                st.subheader("Accident Severity vs. Month")
                st.altair_chart(month_chart, use_container_width=True)

                severity_intersection = pd.crosstab(filtered_data['Accident Severity'],filtered_data['numIntersections'].sum())
                st.subheader("Accident Severity vs. Intersections")
                st.bar_chart(severity_intersection)

                # Speed vs Accident Severity
                st.subheader("Speed vs. Accident Severity")
                speed_severity = combinedData.groupby('Accident Severity')['SPEED'].mean()
                st.bar_chart(speed_severity)

            st.markdown("### Detailed Data View")
            filtered_data_display = filtered_data.copy()
            filtered_data_display['geometry'] = filtered_data_display['geometry'].astype(str)  # Convert geometry to string
            st.dataframe(filtered_data_display)
        else:
            st.warning("No data in this map area. Please zoom or pan to another location.")
    else:
        st.error("Please adjust the map to display data.")
else:
    st.info("Interact with the map above to display data.")
