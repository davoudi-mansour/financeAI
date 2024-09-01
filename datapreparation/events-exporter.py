import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from datetime import datetime, date

START_DATE = date(2023, 1, 1)
END_DATE = date(2023, 12, 31)

YEAR = START_DATE.year

RESULT_PATH_JSON = "./Events/events" + str(YEAR) + ".json"
RESULT_PATH_CSV = "./Events/events" + str(YEAR) + ".csv"

start = START_DATE.strftime("%Y%m%d")
end = END_DATE.strftime("%Y%m%d")

url = f"http://calendar.fxstreet.com/EventDateWidget/GetMini?culture=en-US&view=range&start={start}&end={end}"\
      f"&timezone=UTC&columns=date%2Ctime%2Ccountry%2Ccountrycurrency%2Cevent%2Cconsensus%2Cprevious%2Cvolatility" \
      f"%2Cactual&showcountryname=true&showcurrencyname=true&isfree=true&_=1455009216444 "

response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

f = open("soup.txt", "a")
f.write(str(soup))
f.close()

# Find the table with the specified ID
table = soup.find('table', {'id': 'fxst-calendartable'})

# Extract table rows
rows = table.find_all('tr')

data_list = []
current_date = None

for row in rows:
    if 'fxst-dateRow' in row.get('class', []):

        date = str(YEAR) + ' ' + row.find('td').text.strip()
        current_date = datetime.strptime(date, '%Y %A, %b %d').strftime('%Y-%m-%d')
    else:
        columns = row.find_all('td')
        if len(columns) >= 7 and current_date:
            data = {
                "Date": current_date,
                "Time": columns[0].text.strip(),
                "Country": columns[1].text.strip(),
                "Currency": columns[2].text.strip(),
                "Event": columns[3].text.strip(),
                "Volatility": columns[4].text.strip(),
                "Actual": columns[5].text.strip(),
                "Consensus": columns[6].text.strip(),
                "Previous": columns[7].text.strip()
            }
            data_list.append(data)

# Convert the list of dictionaries to JSON format and save
json_data = json.dumps(data_list, indent=4)
with open(RESULT_PATH_JSON, 'w') as fp:
    fp.write(json_data)

# Convert the list of dictionaries to CSV format and save
df = pd.DataFrame(data_list)
df.to_csv(RESULT_PATH_CSV, index=False)
