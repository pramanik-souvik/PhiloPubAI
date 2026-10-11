# Data

## `survey_harmonised.csv`

A structured 15-item questionnaire administered in Dhaka, Bangladesh to six occupational groups. Each group was collected as a separate spreadsheet ("source sheet") and then harmonised into a common schema: unified column names and category spellings, header-leak and empty rows removed. One row is one respondent (n = 7,599). There are no names or contact details.

| Column | Type | Description / values |
|---|---|---|
| `row_id` | int | Record identifier (1–7,599) |
| `occupation` | category | Source sheet: Bus Driver/Private Car Driver (AC); Bus Driver (Non-AC); Rickshaw Puller/Biker; Traffic Cop/Fire Fighter/Guard; Business Owner; Student |
| `living_area` | category | Residential zone (54 zones) |
| `work_area` | category | Workplace zone (67 zones) |
| `stay_home_hours` | numeric | Average hours per day at home (4–20) |
| `stay_office_hours` | numeric | Average hours per day at the workplace (1–12) |
| `travel_hours` | numeric | Average hours per day travelling (0.01–20) |
| `travel_timing` | category | Morning, Noon, Afternoon, Evening, Night |
| `travel_vehicle` | category | 15 modes: Bus, Rickshaw, Bike, Motorbike, Bicycle, Walking, CNG, CNG Auto-rickshaw, Leguna, Tempo, Car, Private Car, Uber, Train, Metro |
| `ac_service` | category | Access to air-conditioning: Yes, No, Unknown |
| `work_days` | numeric | Working days per week (2–7) |
| `house_type` | category | Tin Shed, Semi-pucca, Pucca, Independent, Flat, Flat-Rent, Apartment, Open House, Shared Dorm |
| `age` | numeric | Years (13–70) |
| `gender` | category | Male, Female, Intersex |
| `smoker_status` | category | Smoker, Non-smoker |
| `lung_disease` | category | Self-reported: No Disease, Asthma, Tuberculosis, Bronchitis, Chronic Bronchitis, Bronchiectasis, Allergy, Pneumonia, Fungal Lung Infections, Lung Cancer, Mesothelioma, Cystic Fibrosis, Sarcoidosis, Unknown |

### Coding notes

- **`Unknown` in `lung_disease` (n = 1,756)** stands for a blank field in the source sheets. The data collectors confirmed that a blank means the respondent reported **no lung disease**. The analysis codes these records as *No Disease*. All of them are in the five worker sheets, which had no explicit "no disease" code.
- **`Bronchiectasis` among students (n = 65)** comes from a raw code `Y` in the student sheet. The analysis recodes these records as *Unspecified* disease.
- **`ac_service = Unknown`** occurs only in the business-owner sheet.
- **Known artefact:** in the business-owner sheet, every smoker reports a disease and every non-smoker reports none. The analysis detects and reports this pattern; see the paper and `results/tab_artefacts.tex`.

## `area_pm25_lookup.csv`

Annual mean (2024) PM2.5 in µg/m³ from the CAMS reanalysis, retrieved through the Open-Meteo air-quality API for zone centroids. Columns: `area`, `lat`, `lon`, `pm25_avg`, `shared_grid_cell`. Only 20 of 77 zones could be geocoded, and because of the roughly 40 km reanalysis grid these take just two distinct values. The paper uses this file only to assess ambient-exposure resolution and coverage (indicators E4 and J2).
