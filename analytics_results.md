# NYC Yellow Taxi — Cloud Analytics Results

Generated from the EMR Serverless pipeline (`cloud/02_preprocess_cloud.py` →
`cloud/03_analytics_cloud.py`) run on 2026-07-30 against the full multi-year
dataset (2019–2022) in `s3://ds-cs675-cweng-workspace/data/taxi/`.

Raw query outputs (Parquet, written by the Spark job) were downloaded from
`s3://ds-cs675-cweng-workspace/output/results/` and are checked in as CSV
under [`results/`](results/) for easy inspection/diffing.

## Headline numbers

- **Total trips (post-cleaning):** 177,174,900
- **Total revenue (fare_amount):** $2,399,047,291.20
- **Average fare per trip:** $13.54

## Q1 — Demand by hour and day of week

`results/q1_hourly_demand.csv` — trip counts for all 168 (hour × day-of-week) buckets.
(`day_of_week`: 1=Sun, 2=Mon, ... 7=Sat, matching Spark's `dayofweek()`.)

- **Weekday trips (Mon–Fri):** 130,706,588 (74% of all trips)
- **Weekend trips (Sat/Sun):** 46,468,312 (26%)
- **Top weekday hours:** 18:00 (9.18M), 17:00 (8.47M), 19:00 (8.34M) — a single
  evening peak rather than the classic AM/PM commute double-hump, suggesting
  taxi demand skews toward evening/dinner travel more than office commuting.
- **Top weekend hours:** 18:00 (2.91M), 17:00 (2.88M), 15:00 (2.81M) — same
  evening-leaning shape as weekdays, just at ~3x lower volume per hour.

## Q2 — Revenue by pickup zone

`results/q2_zone_revenue.csv` — all 264 zones, sorted by total revenue.

Top 5 pickup zones by total revenue (zone names from the [NYC TLC zone
lookup](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv)):

| PULocationID | Zone | Borough | Total Revenue | Avg Fare | Trip Count |
|---|---|---|---|---|---|
| 132 | JFK Airport | Queens | $279,797,900 | $45.73 | 6,118,817 |
| 138 | LaGuardia Airport | Queens | $132,983,700 | $30.98 | 4,292,378 |
| 161 | Midtown Center | Manhattan | $82,492,980 | $11.78 | 7,003,527 |
| 237 | Upper East Side South | Manhattan | $77,854,430 | $9.56 | 8,140,446 |
| 186 | Penn Station/Madison Sq West | Manhattan | $74,479,570 | $12.11 | 6,151,179 |

The two airport zones stand out sharply: far fewer trips than the
high-volume Manhattan zones (161, 237, 236) but 3–4x the average fare, since
airport trips are long and/or flat-rate. They generate the most total
revenue despite not being the busiest pickup points.

## Q3 — Tipping behavior by trip distance

`results/q3_tipping_by_distance.csv`

| Distance bucket | Avg tip % | Avg tip $ | Trip count |
|---|---|---|---|
| short (<1mi) | 22.42% | $1.23 | 39,753,936 |
| medium (1–10mi) | 18.86% | $2.24 | 125,444,594 |
| long (>10mi) | 16.69% | $6.87 | 11,976,370 |

Tip *percentage* falls as trips get longer, even though tip *dollar amount*
rises — riders tip a smaller share of a larger fare on long trips. Short
trips get the highest percentage tips but the lowest dollar amount.

## Q4 — Fare efficiency by trip distance

`results/q4_fare_per_mile.csv`

| Distance bucket | Avg fare/mile | Avg fare | Avg distance | Trip count |
|---|---|---|---|---|
| short (<1mi) | $19.10 | $5.83 | 0.68 mi | 39,753,936 |
| medium (1–10mi) | $5.20 | $12.70 | 2.76 mi | 125,444,594 |
| long (>10mi) | $3.07 | $47.96 | 34.83 mi | 11,976,370 |

Confirms the expected base-fare effect: short trips cost ~6x more per mile
than long trips because the flat base fare dominates the total for short
distances, while long trips (often airport runs, given Q2) amortize it over
many more miles.

## Q5 — Demand by Weather Condition (local sample)

`results/q5_weather_demand.csv` — computed on the **local Jan 2022 sample** (not the full cloud-scale run), joined against NOAA daily weather for NYC Central Park.

| Weather condition | Trip count | Avg fare | Avg distance |
|---|---|---|---|
| clear | 1,670,147 | $12.56 | 5.07 mi |
| rain | 407,351 | $13.68 | 6.49 mi |
| snow | 345,742 | $12.72 | 6.09 mi |

Demand drops sharply once weather turns bad: 69% of trips happen on clear days versus 17% on rain and 14% on snow — fewer people are out and about to hail a cab. But the trips that do happen skew longer and pricier: average fare rises from $12.56 on clear days to $13.68 (rain) and $12.72 (snow), and average distance jumps from 5.07mi to 6.49mi/6.09mi. That's consistent with riders reserving cabs for longer or harder-to-walk trips in bad weather rather than the short local hops that pad out clear-day volume.
