import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# Real Bengaluru locations with real lat/lon
locations = {
    'Silk Board': (12.9173, 77.6226, 1.6),  # 1.6 = very high traffic factor
    'Hebbal': (13.0358, 77.5970, 1.4),
    'Whitefield': (12.9698, 77.7500, 1.3),
    'Electronic City': (12.8456, 77.6603, 1.3),
    'MG Road': (12.9757, 77.6058, 1.1),
    'Koramangala': (12.9279, 77.6271, 1.2),
    'Bellandur': (12.9259, 77.6765, 1.4),
    'Majestic': (12.9767, 77.5713, 1.2),
    'HSR Layout': (12.9116, 77.6473, 1.0),
    'Yelahanka': (13.1007, 77.5963, 0.8),
}

rows = []
# 30 days of data
dates = pd.date_range('2026-10-15', periods=30, freq='D')

for date in dates:
    is_weekend = date.day_of_week >= 5
    for loc, (lat, lon, factor) in locations.items():
        # 4 time slots per day per location = 1200 rows
        for hour in [8, 11, 17, 20]:
            # Peak hours: 8-10 AM and 5-8 PM
            if hour in [8, 17]:
                peak = 1.7
            elif hour == 20:
                peak = 1.1
            else:
                peak = 0.8

            if is_weekend:
                peak *= 0.7

            # Random variation
            noise = np.random.uniform(0.8, 1.2)

            cars = int(180 * factor * peak * noise)
            bikes = int(220 * factor * peak * noise)
            buses = int(35 * factor * peak * noise)
            autos = int(90 * factor * peak * noise)

            total = cars + bikes + buses + autos

            # Speed goes down when traffic goes up
            speed = max(8, 55 - (total / 18) + np.random.uniform(-3, 3))
            speed = round(speed, 1)

            weather = np.random.choice(['Clear', 'Clear', 'Clear', 'Rain', 'Cloudy'], p=[0.6, 0.1, 0.1, 0.1, 0.1])
            if weather == 'Rain':
                speed -= 5
                total = int(total * 1.15)

            # Congestion Level
            if total > 800:
                congestion = 'High'
            elif total > 500:
                congestion = 'Medium'
            else:
                congestion = 'Low'

            rows.append([
                date.strftime('%Y-%m-%d'),
                f"{hour:02d}:00",
                hour,
                loc, lat, lon,
                cars, bikes, buses, autos,
                total, speed, weather,
                'Weekend' if is_weekend else 'Weekday',
                congestion
            ])

df = pd.DataFrame(rows, columns=[
    'Date', 'Time', 'Hour', 'Location', 'Latitude', 'Longitude',
    'Cars', 'Bikes', 'Buses', 'Autos',
    'Total_Vehicles', 'Avg_Speed', 'Weather', 'Day_Type', 'Congestion_Level'
])

df.to_csv('bengaluru_traffic.csv', index=False)
print("Done! Rows:", len(df))
print(df.head(10))
print(df['Congestion_Level'].value_counts())
print(df['Location'].value_counts())
