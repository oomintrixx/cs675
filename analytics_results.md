# NYC Yellow Taxi — Cloud Analytics Results

Generated from the EMR Serverless pipeline (`cloud/02_preprocess_cloud.py` →
`cloud/03_analytics_cloud.py`) run on 2026-08-04 against the full multi-year
dataset (2019–2022) in `s3://ds-cs675-cweng-workspace/data/taxi/`. All five
queries — including Q5's weather join — ran at full cloud scale in this run.

Raw query outputs (Parquet, written by the Spark job) were downloaded from
`s3://ds-cs675-cweng-workspace/output/results/` and are checked in as CSV
under [`results/`](results/) for easy inspection/diffing.

## Headline numbers

- **Total trips (post-cleaning):** 177,171,999
- **Total revenue (fare_amount):** $2,399,047,291.20
- **Average fare per trip:** $13.54

## Q1 — Demand by hour and day of week

`results/q1_hourly_demand.csv` — trip counts for all 168 (hour × day-of-week) buckets.
(`day_of_week`: 1=Sun, 2=Mon, ... 7=Sat, matching Spark's `dayofweek()`.)

- **Weekday trips (Mon–Fri):** 130,704,405 (74% of all trips)
- **Weekend trips (Sat/Sun):** 46,467,594 (26%)
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
| 132 | JFK Airport | Queens | $279,797,933 | $45.73 | 6,118,438 |
| 138 | LaGuardia Airport | Queens | $132,983,656 | $30.98 | 4,292,250 |
| 161 | Midtown Center | Manhattan | $82,492,983 | $11.78 | 7,003,430 |
| 237 | Upper East Side South | Manhattan | $77,854,434 | $9.56 | 8,140,361 |
| 186 | Penn Station/Madison Sq West | Manhattan | $74,479,568 | $12.11 | 6,151,083 |

The two airport zones stand out sharply: far fewer trips than the
high-volume Manhattan zones (161, 237, 236) but 3–4x the average fare, since
airport trips are long and/or flat-rate. They generate the most total
revenue despite not being the busiest pickup points.

## Q3 — Tipping behavior by trip distance

`results/q3_tipping_by_distance.csv`

| Distance bucket | Avg tip % | Avg tip $ | Trip count |
|---|---|---|---|
| short (<1mi) | 22.42% | $1.23 | 39,752,916 |
| medium (1–10mi) | 18.86% | $2.24 | 125,443,296 |
| long (>10mi) | 16.69% | $6.87 | 11,975,787 |

Tip *percentage* falls as trips get longer, even though tip *dollar amount*
rises — riders tip a smaller share of a larger fare on long trips. Short
trips get the highest percentage tips but the lowest dollar amount.

## Q4 — Fare efficiency by trip distance

`results/q4_fare_per_mile.csv`

| Distance bucket | Avg fare/mile | Avg fare | Avg distance | Trip count |
|---|---|---|---|---|
| short (<1mi) | $19.10 | $5.83 | 0.68 mi | 39,752,916 |
| medium (1–10mi) | $5.20 | $12.70 | 2.76 mi | 125,443,296 |
| long (>10mi) | $3.07 | $47.96 | 34.83 mi | 11,975,787 |

Confirms the expected base-fare effect: short trips cost ~6x more per mile
than long trips because the flat base fare dominates the total for short
distances, while long trips (often airport runs, given Q2) amortize it over
many more miles.

## Q5 — Demand by weather condition

`results/q5_weather_demand.csv` — the cross-source join: all 177M+ cleaned
trips (2019–2022) joined against 4 years of NOAA daily weather for NYC
Central Park (`USW00094728`), broadcasting the tiny weather side against the
taxi fact table. A day is bucketed by any measurable precipitation
(`PRCP > 0` or `SNOW > 0`); snow takes priority when both are non-zero on
the same day.

| Weather condition | Trip count | Share | Avg fare | Avg distance |
|---|---|---|---|---|
| clear | 110,718,538 | 62.5% | $13.57 | 4.47 mi |
| rain | 61,412,663 | 34.7% | $13.56 | 4.52 mi |
| snow | 5,039,027 | 2.8% | $12.65 | 3.62 mi |

At full 4-year scale, weather has a much smaller effect on fare and distance
than a first look at a single snowy month might suggest: avg fare is
essentially flat across clear ($13.57) and rain ($13.56) days, and actually
**lower** on snow days ($12.65, -6.8% vs. clear) — the opposite direction
from what a naive "bad weather → higher fares" intuition would predict.
Average distance follows the same pattern: snow trips run *shorter* (3.62 mi
vs. 4.47 mi clear, -19%), not longer. A plausible explanation, consistent
with Q2: snowstorms likely suppress the longest, highest-fare trips first
(airport runs, cross-borough trips) more than they suppress short local
hops, pulling both the average fare and average distance down on snow days
rather than up.

Trip volume itself: clear days account for the large majority of trips
(62.5%), with rain a substantial secondary share (34.7%) — unsurprising
given NYC gets far more rainy days than snow days across a full year — and
snow a small tail (2.8%), consistent with snow being both rarer and
typically shorter-lived per storm than multi-day rain systems.

**Earlier local-sample estimate, for comparison:** before this cloud-scale
run, Q5 was first validated on the local Jan 2022 dev slice only (31 days:
20 clear, 6 rain, 5 snow — including the Jan 29, 2022 NYC blizzard). That
one-month sample happened to have an unusually snow-heavy month and showed
the opposite direction on fare/distance (snow trips *higher* fare/distance
than clear), which the cloud-scale run above does not confirm — a good
illustration of why a 31-day local sample from one unusual month shouldn't
be treated as a reliable signal on its own, and why running the full
4-year, 177M-row join was worth doing rather than stopping at the local
result.
