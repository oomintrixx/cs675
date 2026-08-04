# NYC Yellow Taxi — Cloud Analytics Results

Generated from the EMR Serverless pipeline (`cloud/02_preprocess_cloud.py` →
`cloud/03_analytics_cloud.py`) run on 2026-07-30 against the full multi-year
dataset (2019–2022) in `s3://ds-cs675-cweng-workspace/data/taxi/`.

Raw query outputs (Parquet, written by the Spark job) were downloaded from
`s3://ds-cs675-cweng-workspace/output/results/` and are checked in as CSV
under [`results/`](results/) for easy inspection/diffing — except
`q5_weather_demand.csv`, which is a local run — see the Q5 section below.

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

## Q5 — Demand by weather condition

`results/q5_weather_demand.csv` — computed on the **local Jan 2022 sample** (not the full cloud-scale run), joined against NOAA daily weather for NYC Central Park. A day is bucketed by any measurable precipitation (`PRCP > 0` or `SNOW > 0`); snow takes priority when both are non-zero on the same day.

| Weather condition | Trip count | Avg fare | Avg distance |
|---|---|---|---|
| clear | 1,670,147 | $12.56 | 5.07 mi |
| rain | 407,351 | $13.68 | 6.49 mi |
| snow | 345,742 | $12.72 | 6.09 mi |

January 2022 had 20 clear days, 6 rain days, and 5 snow days, so the raw
trip-count shares above (69% / 17% / 14%) mostly just mirror how many days
of each type occurred — they are **not** good evidence of a demand effect on
their own. Normalizing to trips/day isolates the actual effect: **~83,500
trips/day on clear days vs. ~67,900/day on rain (-19%) and ~69,100/day on
snow (-17%)**. Daily volume genuinely drops on bad-weather days, just by
less than the raw percentages suggest.

The fare difference is real and robust: avg fare rises from $12.56 (clear)
to $13.68 (rain, +8.9%) and $12.72 (snow, +1.3%) — trustworthy because
`fare_amount` is capped at $500 in preprocessing (`remove_outliers()` in
`work/preprocess_steps.py`), so a handful of bad rows can't skew it.

The average-*distance* numbers above are much less trustworthy, though:
`trip_distance` is **not** capped in preprocessing, and the cleaned data
still contains a few GPS/meter-error trips up to 306,159 mi. The median
trip distance is far more stable across buckets — 1.73 mi (clear), 1.89 mi
(rain), 1.80 mi (snow) — than the 5.07/6.49/6.09 mi means. Excluding the
handful of trips over 100 mi (81 clear, 23 rain, 30 snow — a tiny fraction
of ~2.4M trips), the mean drops to 3.06 mi (clear), 3.60 mi (rain), and
3.12 mi (snow): the rain increase survives (+17.8%, consistent with the
median's smaller but real rain uptick) but the apparent snow effect nearly
vanishes (+2.2%, well within noise). In short: rain trips really do run a
bit longer on average, but the "snow trips are longer" claim in the raw
means is an artifact of a few outlier rides, not a real weather effect.
