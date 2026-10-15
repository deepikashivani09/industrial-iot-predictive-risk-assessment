from influxdb_client import InfluxDBClient
import pandas as pd

# -----------------------------
# InfluxDB Configuration
# -----------------------------
url = "http://localhost:8086"

token = "INFLUXDB_TOKEN"

org = "organisation name"

bucket = "industrial_data"


client = InfluxDBClient(
    url=url,
    token=token,
    org=org
)

query_api = client.query_api()

query = f'''
from(bucket: "{bucket}")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "industry")
  |> last()
'''

tables = query_api.query(query)

data = {}

for table in tables:
    for record in table.records:
        data[record.get_field()] = record.get_value()

print("\n========== LIVE SENSOR DATA ==========\n")

print(data)

client.close()