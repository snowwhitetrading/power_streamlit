# Power Dashboard — MongoDB Data Guide (for the Frontend)

This document describes **how the UI should read data from MongoDB** and **what each chart is meant to show**.
It is the contract between the data pipeline (`processor_data.py`, `upload_csv.py`) and the frontend.

---

## 1. Where the data lives

| | |
|---|---|
| **Database** | `dc_commodity` |
| **Collection — charts** (Tabs 1–6) | `ThiTruongDien` |
| **Collection — news** (Tab 7) | `TinNganhDien` |

Every **chart** (Tabs 1–6) reads from `ThiTruongDien`; you select its data by filtering on the `dataset`
field (the *source* name, e.g. `fetch_volume_source_monthly`). The **News** tab is the one exception — it
reads article-style records from `TinNganhDien` (see Tab 7).

### Data flow

```
CSV files in data/industries_data/   ──upload_csv.py───▶   MongoDB (dc_commodity.ThiTruongDien)
              data/companies_data/
   fetch_*   = auto-scraped (processor_data.py)
   manual_*  = manually maintained spreadsheets
   fact_*    = static reference tables
   company_* = per-company time-series

JSON feeds in data/                  ──upload_json.py──▶   MongoDB (dc_commodity.TinNganhDien)
   fetch_documents_companies.json  (IR disclosures  → type "document")
   fetch_news_companies.json       (press coverage  → type "news")
   fetch_press_companies.json      (company releases → type "press")
```

`upload_csv.py` normalizes **every** CSV into one common document shape. The CSV **file name (without
extension)** becomes the `dataset` value — that is the *source* name you query by. `upload_json.py` does
the same for the three JSON feeds into `TinNganhDien`.

### Source naming convention

| Prefix | Meaning | Updated by |
|---|---|---|
| `fetch_` | Scraped automatically on a schedule | `processor_data.py` |
| `manual_` | Maintained by hand (analyst spreadsheet) | uploaded CSV |
| `fact_`  | Static reference / master data | uploaded CSV |

---

## 2. Document schema

Each document = one `(dataset, date, object)` combination.

```jsonc
{
  "_id": "…",                       // stable hash of (dataset, date, object)
  "dataset": "fetch_volume_source_monthly",   // ← the SOURCE name you filter on
  "object": "thuy_dien",            // the entity/series within the dataset (group key)
  "date": ISODate("2025-06-01"),    // null for date-less reference tables
  "date_text": "2025-06-01",        // raw date string as stored
  "values": { "metric_a": 123.4, "metric_b": 56.7 },   // numeric payload
  "units":  { "metric_a": "MW",   "metric_b": "%"  },  // unit per metric
  "source_file": "fetch_volume_source_monthly.csv",
  "updated_at": ISODate("…")
}
```

**Reading rule of thumb**

1. Filter by `dataset` (the source).
2. Group / split series by `object`.
3. Read the numbers from `values.<metric>`; read its unit from `units.<metric>`.
4. Use `date` for the time axis.

```js
// Generic query pattern
db.ThiTruongDien.find({ dataset: "<source>", date: { $gte: from, $lte: to } })
                .sort({ date: 1 })
```

> The frontend is responsible for **time aggregation** (daily → monthly/quarterly/yearly) and
> **derived metrics** (YoY, averages, sums). The collection stores the finest granularity available.

---

## 3. Charts by tab

Legend for each entry: **source** = `dataset` value · **chart** = visual · **unit**.

### Tab 1 — Government (static `fact_*`)

| Chart | source (`dataset`) | chart type | unit |
|---|---|---|---|
| Revised PDP8 Capacity | `fact_pdp8_capacity` | stacked bar | MW |
| Revised PDP8 Capex | `fact_pdp8_capex` | stacked bar for `2026-2030` & `2031-2035` (split **State** vs **Socialized**); simple bar for `2020-2025` (show **Total**) | USDbn |
| Price Framework | `fact_price_framework` | simple bar | VND/kWh |
| Revised PDP Plants | `fact_pdp8_plant` | table with dropdown (aggregate capacity by **power type / source**) | — |

- `fact_pdp8_capacity` — `object` = power type (`Total`, `Hydro Power`, …); `values` = capacity per
  scenario column (`Capacity 2025`, `Capacity 2030 Thấp`, `Capacity 2030 Cao`); `date` = year-end.
- `fact_pdp8_capex` — dimensions: **Phase**, **Investment Type**, and Total/State/Socialized capital.
- `fact_price_framework` — dimensions: **Power Source**, **Region**, price in VND/kWh (+ US cent/kWh).
- `fact_pdp8_plant` — per-project rows: Project, Type, Capacity (MW), Expected Operation, Investor, Location.

> ⚠️ `fact_pdp8_capex`, `fact_pdp8_plant`, and `fact_price_framework` are **dimensional tables**
> (their key columns are text and they have no time axis). If you need the raw rows intact, read them
> from a dedicated representation rather than relying on the generic `object/values` split — see
> §5 Caveats.

### Tab 2 — Weather

| Chart | source (`dataset`) | chart type | unit |
|---|---|---|---|
| ENSO probability | `fetch_enso_probability_monthly` | stacked bar (option: view by issue date) | % |
| ENSO condition | `fetch_hydrology_current_monthly` | bar — **red** = El Niño, **blue** = La Niña, **grey** = Neutral | — |
| Vietnam temperature | `fetch_weather_temperature_monthly` | line (option: average daily/monthly/yearly; one line per city Sài Gòn / Đà Nẵng / Hà Nội; legend = year) | °C |

- `fetch_enso_probability_monthly` — `values` = `La Nina prob`, `Neutral prob`, `El Nino prob`;
  `date` = forecast issue month. (`season` is the forecast horizon — see Caveats.)
- `fetch_hydrology_current_monthly` — `values.oni_value` (ONI). Classify the bar color from the ONI value:
  `> 0.5` El Niño (red), `< -0.5` La Niña (blue), otherwise Neutral (grey). `enso_type` text is also in the CSV.
- `fetch_weather_temperature_monthly` — `object` = city (`Ho_Chi_Minh`, `Da_Nang`, `Ha_Noi`);
  `values.temp_mean`. Daily granularity — aggregate to monthly/yearly on the client.

### Tab 3 — Cost

| Chart | source (`dataset`) | chart type | unit |
|---|---|---|---|
| Coal cost | `manual_coal_cost_monthly` | line | VND/ton |
| Gas cost | `manual_gas_cost_monthly` | line | USD/MMBTU |
| Water reservoir | `fetch_hydro_reservoir_monthly` | bar — region YoY growth (x = month/quarter/year, legend = region, with date option) | % |

- `manual_coal_cost_monthly` — `values` = `Coal Cost for Vinh Tan`, `Coal Cost for Mong Duong` (VND/ton).
- `manual_gas_cost_monthly` — `values` = `Phu My Gas Cost`, `Nhon Trach Gas Cost` (USD/MMBTU).
- `fetch_hydro_reservoir_monthly` — `object` = reservoir name; `values.flood_level` (m),
  `values.flood_capacity` (%), `values.plant_throughput` (m³/s). Computation chain for the chart:
  1. monthly average flood level **per reservoir**;
  2. YoY growth of that monthly average per reservoir;
  3. aggregate YoY growth to **region** level (repeat for quarterly and yearly).
  - **Region mapping** is *not* stored on the reservoir documents — join reservoir → region via
    `fact_water_list`.

### Tab 4 — Price

| Chart | source (`dataset`) | chart type | unit |
|---|---|---|---|
| CGM price | `fetch_price_smp_monthly` (SMP) **+** `manual_price_can_annually` (CAN) → **CGM = SMP + CAN** | 4 line charts (North / Central / South / National; option daily/monthly/quarterly/yearly; legend = year) | VND/kWh |
| CAN price | `manual_price_can_monthly` | bar | VND/kWh |
| Retail price | `manual_price_retail_monthly` | line | VND/kWh |

- `fetch_price_smp_monthly` — **nested intraday structure**. `object` = `giaBien`; `values` is keyed by
  half-hour time slot, each holding the four regional SMPs:

  ```jsonc
  "values": {
    "00:00": { "giaBienMB": …, "giaBienMT": …, "giaBienMN": …, "giaBienHT": … },
    "00:30": { … }, …
  }
  ```

  `giaBienMB` = North, `giaBienMT` = Central, `giaBienMN` = South, `giaBienHT` = whole-system (National).
  Compute the **daily average** per region across the time slots, then add the CAN price → CGM.
- `manual_price_can_annually` — `values.CAN price` (VND/kWh), one point per year.
- `manual_price_retail_monthly` — `values.Retail price` (VND/kWh).

> ⚠️ The CAN price chart references `manual_price_can_monthly`, but only `manual_price_can_annually`
> exists today. Confirm whether a monthly CAN source will be added, or whether the bar chart should use
> the annual file.

### Tab 5 — Volume

| Chart | source (`dataset`) | chart type | unit |
|---|---|---|---|
| Volume by source | `fetch_volume_source_monthly` (col `generation_thuong_pham_mkWh`) | bar (aggregate daily/monthly/quarterly/yearly) | million kWh |
| Volume breakdown by source | `fetch_volume_source_monthly` (breakdown cols below) | stacked bar (aggregate daily/monthly/quarterly/yearly) | million kWh |
| Capacity mismatch | supply: `fetch_capacity_mobilized_monthly` (rows `quốc gia_đầu cực`, col `high_season`) · demand: `fetch_volume_source_monthly` (col `max_power_dau_cuc_MW`) | bar (max of daily/monthly/quarterly/yearly) | MW |
| Capacity by source | `fetch_capacity_installed_monthly` (latest day only) | pie + MW note | MW |
| Capacity by region | `fetch_volume_region_monthly` (monthly average) | stacked bar (avg monthly/quarterly/yearly) | MW |

- `fetch_volume_source_monthly` — single `object`; daily docs. Breakdown metrics for the stacked bar:
  `thuy_dien_mkWh`, `nhiet_dien_than_mkWh`, `tuabin_khi_mkWh`, `nhiet_dien_dau_mkWh`, `dien_gio_mkWh`,
  `dmt_trang_trai_mkWh`, `dmt_mai_thuong_pham_mkWh`, `nhap_khau_mkWh`, `khac_mkWh`.
  Also carries `max_power_dau_cuc_MW` (demand peak, used by Capacity mismatch).
- `fetch_capacity_mobilized_monthly` — `object` = source row (e.g. `Quốc gia_Đầu cực`);
  `values.mobilized_high_season` / `values.mobilized_low_season` (MW). Use the `high_season` value as supply.
- `fetch_capacity_installed_monthly` — `object` = power type; `values.installed_capacity` (MW). For the
  pie, take the **latest** `date` only.
- `fetch_volume_region_monthly` — `object` = `congSuat`; `values` = `congSuatMB` (North), `congSuatMT`
  (Central), `congSuatMN` (South), `congSuatHT` (whole system), in MW. Average over the period.

### Tab 6 — Company (per-company; **ticker selector** control)

Unlike Tabs 1–5 (one topic across the whole market), this tab shows **one company at a time**. A
**ticker dropdown** is the control: picking a company points every chart on the tab at that company's
`company_<ticker>_<freq>` dataset. Within each chart, series are split by `object`, numbers come from
`values.<metric>`, and the time axis is `date`.

| Ticker | source (`dataset`) | cadence | `object` means |
|---|---|---|---|
| GEG | `company_geg_quarterly` | quarterly | generation source |
| HDG | `company_hdg_quarterly` | quarterly | business segment / generation source |
| HND | `company_hnd_quarterly` | quarterly | *(single — the ticker itself)* |
| PPC | `company_ppc_quarterly` | quarterly | *(single — the ticker itself)* |
| QTP | `company_qtp_quarterly` | quarterly | *(single — the ticker itself)* |
| PC1 | `company_pc1_quarterly` | quarterly | business segment / generation source |
| REE | `company_ree_quarterly` | quarterly | business segment / generation source |
| PGV | `company_pgv_monthly` | monthly | power plant |
| POW | `company_pow_monthly` | monthly | power plant |

**Metric keys** (`values.<…>`) and units, shared across companies:

| metric | meaning | unit |
|---|---|---|
| `Volume` | output volume | mn kWh |
| `Mobilized` / `Contracted` | mobilized / contracted volume (plant-level companies) | mn kWh |
| `Revenue` | source/plant revenue | VNDbn |
| `net revenue` | business-segment net revenue | VNDbn |
| `gross profit` | business-segment gross profit | VNDbn |
| `NPAT` | net profit after tax (REE only) | VNDbn |
| `ASP` | average selling price | VND/kWh |

> Not every `object` carries every metric: **generation sources** hold `Volume`/`Revenue`/`ASP`, while
> **business segments** hold `net revenue`/`gross profit` (or `NPAT`). Each chart below names the exact
> metric + object set it draws from, verified against the current data.

#### GEG — `company_geg_quarterly` · object = `Solar`, `Wind`, `Hydro`, `Others`
| Chart | type | metric | objects |
|---|---|---|---|
| Revenue breakdown | stacked bar | `Revenue` | Solar, Wind, Hydro, Others |
| Volume breakdown | stacked bar | `Volume` | Solar, Wind, Hydro |
| ASP | line (one per source) | `ASP` | Solar, Wind, Hydro |

> `Others` carries `Revenue` only (no `Volume`/`ASP`).

#### HDG — `company_hdg_quarterly`
Generation sources (`Volume`/`Revenue`/`ASP`): **Hydro, Solar, Wind**. Business segments
(`net revenue`/`gross profit`): **Power, Real estate, Construction, Leasing & Hotel, Others**.

| Chart | type | metric | objects |
|---|---|---|---|
| Revenue breakdown | stacked bar | `net revenue` | Power, Real estate, Construction, Leasing & Hotel, Others |
| Gross profit breakdown | stacked bar | `gross profit` | same 5 segments |
| Volume breakdown | stacked bar | `Volume` | Hydro, Solar, Wind |
| ASP | line (one per source) | `ASP` | Hydro, Solar, Wind |

#### HND / PPC / QTP — `company_{hnd,ppc,qtp}_quarterly` · single object = the ticker
| Chart | type | metric | unit |
|---|---|---|---|
| Revenue | simple bar | `Revenue` | VNDbn |
| Volume | simple bar | `Volume` | mn kWh |
| ASP | line | `ASP` | VND/kWh |

#### PC1 — `company_pc1_quarterly`
Generation sources (`Volume`/`Revenue`/`ASP`): **Hydro, Wind**. Business segments
(`net revenue`/`gross profit`): **Production, Power, Construction, Real estate sales, Real estate leasing, Niken, Nomura IP, Others**.

| Chart | type | metric | objects |
|---|---|---|---|
| Revenue breakdown (by segment) | stacked bar | `net revenue` | the 8 business segments |
| Gross profit (by segment) | stacked bar | `gross profit` | the 8 business segments |
| Revenue breakdown (by power) | stacked bar | `Revenue` | Hydro, Wind |
| Volume breakdown (by power) | stacked bar | `Volume` | Hydro, Wind |
| ASP | line (one per source) | `ASP` | Hydro, Wind |

#### REE — `company_ree_quarterly`
Generation sources (`Volume`/`Revenue`/`ASP`): **Hydro, Solar, Wind** (plus **Thermal**, `Volume` only).
Business segments (`net revenue` and/or `NPAT`): **Power, Water, M&C, Office leasing & Real estate, Thermal & Others, Others**.

| Chart | type | metric | objects |
|---|---|---|---|
| Revenue breakdown (by segment) | stacked bar | `net revenue` | business segments |
| Gross profit (by segment) | stacked bar | `NPAT` ⚠️ | business segments |
| Revenue breakdown (by power) | stacked bar | `Revenue` | Hydro, Solar, Wind |
| Volume breakdown (by power) | stacked bar | `Volume` | Hydro, Solar, Wind, Thermal |
| ASP | line (one per source) | `ASP` | Hydro, Solar, Wind |

> ⚠️ REE stores **`NPAT`**, not `gross profit`. The "gross profit (by segment)" chart must read `NPAT`
> today; add a gross-profit column to the source CSV if true gross profit is required (see §5).

#### PGV — `company_pgv_monthly` · object = power plant (`Volume` = mobilized volume)
Plants: **Phu My, Buon Kuop, Vinh Tan 2, Mong Duong, BTP, NBP, Associates (VSH+TBC+S3A)**;
**`PGV` = company total**.

| Chart | type | metric | objects |
|---|---|---|---|
| Revenue breakdown (by plant) | stacked bar | `Revenue` | per plant (exclude the `PGV …` total) |
| Mobilized volume breakdown (by plant) | stacked bar | `Volume` | per plant |
| ASP | line (one per plant) | `ASP` | per plant |

#### POW — `company_pow_monthly` · object = power plant
Plants: **Ca Mau, Vung Ang, Hua Na, Dak Drinh, Nhon Trach 1, Nhon Trach 2 (NT2), Nhon Trach 3 & 4**;
**`POW` = company total**.

| Chart | type | metric | objects |
|---|---|---|---|
| Revenue breakdown (by plant) | stacked bar | `Revenue` | per plant |
| Mobilized volume breakdown (by plant) | stacked bar | `Mobilized` | per plant |
| Contracted volume breakdown (by plant) | stacked bar | `Contracted` | per plant |
| ASP | line (one per plant) | `ASP` | per plant |

> The `POW` object is the company total (`Mobilized`/`Contracted`/`ASP`, no `Revenue`); `Nhon Trach 3 & 4`
> also has no `Revenue`. Exclude totals from the stacked breakdowns.

#### Company news section (from `TinNganhDien`)

Alongside the charts, each company shows a **News** section pulling the three JSON feeds — disclosures
(`type: "document"`), press coverage (`type: "news"`), and company releases (`type: "press"`) — from the
**`TinNganhDien`** collection (document shape in Tab 7), newest-first by `date`.

The feeds tag companies **differently**, so filtering by the selected ticker needs a small per-company key
map rather than a single symbol: `document`/`press` are matched on `tickers[]`, while `news` is matched on
its `search[]` keyword (often a Vietnamese name or a plant, not the stock code).

| Company | `document` + `press` → `tickers` | `news` → `search` | current items |
|---|---|---|---|
| GEG | `GEG` | `Điện Gia Lai GEG` | 0 doc · 0 press · 1 news |
| HDG | `HDG` | `Tập đoàn Hà Đô` | 0 doc · 0 press · 9 news |
| PC1 | *(none)* | `PC1` | 0 doc · 0 press · 30 news |
| PGV | `PGV` | `EVNGenco3 PGV` | 0 doc · 0 press · 0 news |
| POW | `POW`, `NT2` | `PV POW`, `Điện lực Dầu khí Nhơn Trạch 2`, `Nhơn Trạch 3`, `Nhơn Trạch 4` | 1 doc · 0 press · 17 news |
| REE | `REE` | `REE` | 0 doc · 0 press · 2 news |
| HND | *(none)* | *(none)* | none yet |
| PPC | *(none)* | *(none)* | none yet |
| QTP | *(none)* | *(none)* | none yet |

```js
// News section for POW — union across the three feeds, newest first:
db.TinNganhDien.find({ $or: [
  { type: { $in: ["document", "press"] }, tickers: { $in: ["POW", "NT2"] } },
  { type: "news", search: { $in: ["PV POW", "Điện lực Dầu khí Nhơn Trạch 2", "Nhơn Trạch 3", "Nhơn Trạch 4"] } }
]}).sort({ date: -1 })
```

- **HND / PPC / QTP** have no coverage in the current feeds — their News section will be empty.
- The `news` feed also carries **industry-theme** keywords (`dự án điện`, `giá điện`, `quy hoạch điện`,
  `dầu khí`, `điện khí LNG`, …) that are **not** company-specific — keep those out of a per-company section
  (surface them under Tab 7's general news instead).
- This key map is **data-dependent**; revisit it whenever the feeds add new tickers/keywords.
- The `news` feed is **high-recall** (keeps every relevant article), so a per-company section will include
  low-signal items. Add `is_important: true` (and/or `is_noise: { $ne: true }`) to the `news` branch for a
  clean default, with a "show all" toggle.
- The **current items** counts above are a snapshot of the current **≥ 2026-07-01** window (feeds were
  trimmed to that date); they grow as the pipeline appends new days, so treat them as indicative, not fixed.

#### Water reservoir section (per lake; YTD-average flood level)

Below the charts and news, a company that owns hydro reservoirs gets a **water-reservoir block**: **one bar
chart per lake**. Each chart shows the **year-to-date average flood level** for every year from **2020 →
current** — one bar per year, all cut at the same *Jan 1 → today* window so the years compare like-for-like.
**Under each chart, caption the MW that lake affects and its share of the company's total hydro MW** —
e.g. `55.2 MW · 68% of GEG`.

**Sources & join**
- Lakes owned by the selected company → `fact_water_list` (`ticker` → `lake`).
- **MW per lake** → `fact_water_list` `values.mw` (unit `units.mw` = `MW`) of that `(ticker, lake)` row.
- **Company total MW** (the `%` denominator) → sum of the company's **distinct plant-group** capacities:
  dedupe the `fact_water_list` rows by `plants` (lakes feeding the same plant group count **once**), and
  **include** the `(ngoài file)` capacity. `lake % = lake.mw / companyTotalMW`.
- Flood level → `fetch_hydro_reservoir_monthly`: `object` = reservoir name, `values.flood_level` (m),
  readings from 2020‑01 to now.
- Join on name: `fact_water_list.lake` ⟷ `fetch_hydro_reservoir_monthly.object`.

```js
// Company total MW = sum of DISTINCT plant-group capacities (dedupe by `plants`):
db.ThiTruongDien.aggregate([
  { $match: { dataset: "fact_water_list", ticker: "GEG" } },
  { $group: { _id: "$plants", mw: { $first: "$values.mw" } } },   // one row per plant group
  { $group: { _id: null, totalMW: { $sum: "$mw" } } }
])   // → lake% = that lake's values.mw / totalMW
```

**Per (lake, year) computation**
1. Take `fetch_hydro_reservoir_monthly` docs where `object == <lake>` and `date` falls in
   `[Jan 1 of the year … the same month/day as today]` (the YTD cutoff — today is the reference).
2. Average `values.flood_level` → that year's bar (metres).
3. Repeat for years 2020 … current; **x = year, y = YTD-average flood level**.

```js
// YTD-average flood level per year for one lake (cutoff = today's day-of-year):
db.ThiTruongDien.aggregate([
  { $match: { dataset: "fetch_hydro_reservoir_monthly", object: "Buôn Kuốp" } },
  { $addFields: { y: { $year: "$date" }, doy: { $dayOfYear: "$date" } } },
  { $match: { doy: { $lte: <today's dayOfYear> } } },
  { $group: { _id: "$y", ytdAvgFloodLevel: { $avg: "$values.flood_level" } } },
  { $sort: { _id: 1 } }   // → one bar per year 2020…current
])
```

**Lakes per company** (only lakes that have flood-level data; `(ngoài file)` placeholders excluded):

Each lake below renders one bar chart, captioned `lake (MW · % of company total)`:

| Company | # charts | total MW | lakes — `lake (MW · %)` |
|---|---|---|---|
| GEG | 11 | 81.1 | An Khê (55.2·68.1%), Kanak (55.2·68.1%), Sông Ba Hạ (55.2·68.1%), Sông Hinh (55.2·68.1%), Buôn Kuốp (11.8·14.5%), Buôn Tua Srah (11.8·14.5%), Srêpốk 3 (11.8·14.5%), Đơn Dương (8.1·10.0%), Đại Ninh (8.1·10.0%), Đồng Nai 3 (8.1·10.0%), Đồng Nai 4 (8.1·10.0%) |
| PGV | 10 | 1170 | Buôn Kuốp (280·23.9%), Buôn Tua Srah (86·7.4%), Srêpốk 3 (220·18.8%), Sê San 3A (108·9.2%), Thác Bà (120·10.3%), Sông Hinh (70·6.0%), Thượng Kon Tum (220·18.8%), Vĩnh Sơn A (66·5.6%), Vĩnh Sơn B (66·5.6%), Vĩnh Sơn C (66·5.6%) |
| REE | 8 | 1370.7 | Vĩnh Sơn A (66·4.8%), Vĩnh Sơn B (66·4.8%), Vĩnh Sơn C (66·4.8%), Sông Hinh (70·5.1%), Thượng Kon Tum (227·16.6%), Thác Bà (138.9·10.1%), Thác Mơ (150·10.9%), Sông Ba Hạ (220·16.0%) |
| HDG | 5 | 314 | A Vương (177·56.4%), Sông Bung 2 (177·56.4%), Sông Bung 4 (177·56.4%), Sông Tranh 2 (48·15.3%), Bản Vẽ (89·28.3%) |
| PC1 | 4 | 169 | Tuyên Quang (132·78.1%), Sơn La (30·17.8%), Bản Chát (30·17.8%), Huội Quảng (30·17.8%) |
| POW | 1 | 305 | Trung Sơn (180·59.0%) |
| HND / PPC / QTP | 0 | — | *(thermal — no reservoirs)* |

*(MW = `fact_water_list.values.mw`; total MW = distinct plant-group sum incl. `(ngoài file)`; `%` = MW ÷ total MW.)*

- PGV's `fact_water_list` row spells Vĩnh Sơn as a single lake `Vĩnh Sơn A/B/C`, but the flood-level feed
  stores three reservoirs (`Vĩnh Sơn A` / `B` / `C`) — chart the three (or average them into one).
- **Data gaps** leave some early bars empty: `Thượng Kon Tum` flood level starts **2022**; `Thác Mơ` ends
  **2025‑10** (no 2026 YTD). A year with no readings simply has no bar.
- POW's other reservoir `Đakđrinh` and every `(ngoài file)` lake have **no** entry in the flood-level feed,
  so they are not charted.
- **MW and % are per-lake influence, not additive.** When several lakes feed the same plant group both
  repeat — GEG's An Khê / Kanak / Sông Ba Hạ / Sông Hinh each show `55.2 MW · 68.1%`; HDG's A Vương /
  Sông Bung 2 / Sông Bung 4 each `177 MW · 56.4%`. The per-lake `%` therefore **overlaps and does not sum
  to 100%** — it reads "this reservoir affects X% of the company's hydro capacity". The denominator
  already dedupes shared groups (that's why totals like GEG's 81.1 MW aren't the sum of the chart values).

### Tab 7 — News (⚠️ separate collection: `TinNganhDien`)

This tab does **not** read `ThiTruongDien`. Its documents are article-style records (title / date /
source / url / snippet), loaded from three JSON feeds by `upload_json.py` into
**`dc_commodity.TinNganhDien`**. Pick a feed with the `type` field (or the `dataset` name), then list
newest-first by `date`.

| Feed (`dataset`) | `type` | docs | identify company by | `date_text` format |
|---|---|---|---|---|
| `fetch_documents_companies` | `document` | 8 | `tickers[]` | dd/mm/yyyy |
| `fetch_news_companies` | `news` | 405 (high-recall) | `search[]` (search keyword) | yyyy-mm-dd |
| `fetch_press_companies` | `press` | 11 | `tickers[]` | dd/mm/yyyy |

> Counts are a **snapshot** — all three feeds are currently windowed to **≥ 2026-07-01** (older records
> were trimmed). They grow as the scheduled pipeline appends new days. The `news` feed is also
> **high-recall** (keeps every relevant article, not just analyst-grade), so it stays the largest —
> filter/rank with `is_important` / `is_noise` / `importance_score` (see below).

**Document shape** (`TinNganhDien`):

```jsonc
{
  "_id": "…",                              // hash of (dataset, url|title)
  "dataset": "fetch_news_companies",       // ← the feed
  "type": "news",                          // "document" | "news" | "press"
  "title": "…",                            // headline (always present)
  "url": "https://…",                      // link (always present)
  "date": ISODate("2026-07-05"),           // parsed publish date
  "date_text": "2026-07-05",               // raw date string
  "source": "Tạp chí Nhịp sống thị trường",// publisher — news only
  "snippet": "…",                          // summary — news/press (mostly), rare on documents
  "tickers": ["GAS"],                      // company tags — document & press
  "search": ["PC1"],                       // search keyword that surfaced it — news only
  "importance_score": 9,                   // relevance score — news only (can be negative; higher = more material)
  "is_important": true,                    // analyst-grade event? — news only (ranking flag)
  "is_noise": false,                       // daily price-tracker clutter? — news only (filter flag)
  "matched_topics": ["earnings"],          // topic groups that drove the score — news only
  "query_date": ISODate("2026-07-05"),     // when the news query ran — news only
  "source_file": "fetch_news_companies.json",
  "updated_at": ISODate("…")
}
```

**Field availability by type** (✓ = always, ~ = usually, ✗ = absent):

| field | document | news | press |
|---|---|---|---|
| `title`, `url`, `date`, `date_text` | ✓ | ✓ | ✓ |
| `tickers[]` (company) | ✓ | ✗ (empty) | ✓ |
| `search[]` (keyword) | ✗ | ✓ | ✗ |
| `source` (publisher) | ✗ | ✓ | ✗ |
| `snippet` | ✗ (rare) | ~ (381/405) | ~ (10/11) |
| `importance_score`, `query_date` | ✗ | ✓ | ✗ |
| `is_important`, `is_noise`, `matched_topics` | ✗ | ✓ | ✗ |

- **`document`** & **`press`** are tagged by company via `tickers[]` — filter e.g. `{ type: "press", tickers: "GAS" }`.
- **`news`** is keyword-driven: `tickers[]` is empty; the company/topic it matched is in `search[]`
  (e.g. `["PC1"]`). The feed is **high-recall** — it keeps *every* article that mentions the company or a
  sector keyword (nothing is hard-dropped), so it's up to the UI to rank/filter with the flags:
  - `is_important` — analyst-grade event (matched a financial topic, cleared the score threshold, not noise).
  - `is_noise` — daily commodity price-tracker clutter (`giá gas hôm nay`, …); score is penalized.
  - `importance_score` — relevance rank (can be negative; higher = more material).
  - `matched_topics` — which topic groups fired (`earnings`, `dividend_capital`, `project_contract`, …).

  Suggested default view: `is_important: true` (or at least `is_noise: { $ne: true }`), sorted by
  `date` then `importance_score`, with a "show all" toggle to reveal the full high-recall set.
- Suggested UI: a card/list per item — **title** (→ `url`), **date**, **source** (news), **snippet**,
  and a company/keyword chip from `tickers[]` or `search[]`. Filter by ticker/keyword and by `type`.

```js
// Newest press releases / disclosures for a ticker:
db.TinNganhDien.find({ type: { $in: ["document", "press"] }, tickers: "GAS" }).sort({ date: -1 })
// Newest news for a keyword — full high-recall set, most relevant first:
db.TinNganhDien.find({ type: "news", search: "PC1" }).sort({ date: -1, importance_score: -1 })
// Only analyst-grade news (hide low-signal items and price-tracker noise):
db.TinNganhDien.find({ type: "news", search: "PC1", is_important: true }).sort({ date: -1, importance_score: -1 })
```

> Indexes present: `(dataset, date desc)`, `(type, date desc)`, `(tickers, date desc)`.

---

## 4. Source ↔ chart quick index

| source (`dataset`) | type | used by |
|---|---|---|
| `fact_pdp8_capacity` | fact | Gov: PDP8 Capacity |
| `fact_pdp8_capex` | fact | Gov: PDP8 Capex |
| `fact_price_framework` | fact | Gov: Price Framework |
| `fact_pdp8_plant` | fact | Gov: PDP Plants |
| `fact_water_list` | fact | reservoir → region lookup |
| `fetch_enso_probability_monthly` | fetch | Weather: ENSO probability |
| `fetch_hydrology_current_monthly` | fetch | Weather: ENSO condition |
| `fetch_weather_temperature_monthly` | fetch | Weather: temperature |
| `manual_coal_cost_monthly` | manual | Cost: coal |
| `manual_gas_cost_monthly` | manual | Cost: gas |
| `fetch_hydro_reservoir_monthly` | fetch | Cost: water reservoir |
| `fetch_price_smp_monthly` | fetch | Price: CGM (SMP part) |
| `manual_price_can_annually` | manual | Price: CGM (CAN part) |
| `manual_price_can_monthly` | manual | Price: CAN *(not yet present)* |
| `manual_price_retail_monthly` | manual | Price: retail |
| `fetch_volume_source_monthly` | fetch | Volume: by source, breakdown, mismatch (demand) |
| `fetch_capacity_mobilized_monthly` | fetch | Volume: mismatch (supply) |
| `fetch_capacity_installed_monthly` | fetch | Volume: capacity by source |
| `fetch_volume_region_monthly` | fetch | Volume: capacity by region |
| `company_geg_quarterly` | company | Company: GEG |
| `company_hdg_quarterly` | company | Company: HDG |
| `company_hnd_quarterly` | company | Company: HND |
| `company_ppc_quarterly` | company | Company: PPC |
| `company_qtp_quarterly` | company | Company: QTP |
| `company_pc1_quarterly` | company | Company: PC1 |
| `company_ree_quarterly` | company | Company: REE |
| `company_pgv_monthly` | company | Company: PGV |
| `company_pow_monthly` | company | Company: POW |

**News tab — collection `TinNganhDien`** (not `ThiTruongDien`):

| source (`dataset`) | `type` | used by |
|---|---|---|
| `fetch_documents_companies` | document | News: IR disclosures |
| `fetch_news_companies` | news | News: press coverage |
| `fetch_press_companies` | press | News: company releases |

---

## 5. Caveats / open items

1. **`fetch_price_smp_monthly` is nested** (per-time-slot sub-objects), not a flat `values` map. Average
   across time slots to get a daily figure per region.
2. **Reservoir region** is not on the reservoir documents — join via `fact_water_list`.
3. **`fetch_enso_probability_monthly`**: the generic upload keeps `La/Neutral/El Nino prob` and the
   issue-month `date`, but the `season` horizon column is text and is not split into the `values` map.
   If the stacked bar needs to break out by season, the upload needs a dedicated branch (like
   `fetch_price_smp_monthly` / `fact_pdp8_capacity` already have).
4. **`fact_pdp8_capex` / `fact_pdp8_plant` / `fact_price_framework`** are dimensional/text tables with no
   time axis; the generic `object/values` normalization is lossy for them. Read them as whole rows if you
   need every column.
5. **`manual_price_can_monthly`** referenced by the CAN bar chart does not exist yet (only the annual file).
6. `fetch_coal_vinacomin_monthly` is still scraped (commodity `price`, `percent_change`) but the Cost tab's
   coal chart uses `manual_coal_cost_monthly` instead.
7. **Company `object` mixes two kinds of series** — generation sources (`Volume`/`Revenue`/`ASP`) and
   business segments (`net revenue`/`gross profit`/`NPAT`). Read the metric that matches the object set;
   don't expect `Volume` on a business segment, or `net revenue` on a generation source.
8. ~~PGV objects carry a trailing `" Monthly"`~~ **Fixed** — `upload_csv.py` now strips the cadence word
   from company objects (`strip_cadence_word`), so PGV plants read as `Phu My`, `Buon Kuop`, `PGV`, etc.
9. **REE has `NPAT`, not `gross profit`.** HDG and PC1 store `gross profit`; REE's profit-by-segment chart
   must fall back to `NPAT` until a gross-profit column is added to `company_ree_quarterly.csv`.
10. **Company-total objects** — `PGV` (in PGV) and `POW` (in POW) are the whole-company aggregates, and a
    few plant objects miss `Revenue` (e.g. POW's `Nhon Trach 3 & 4`). Exclude totals from stacked
    breakdowns so they don't double-count.
