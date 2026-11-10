# Water Potability Datasets: Nepal and Supporting Sources

A reference list of datasets for a water potability classification project (Logistic Regression, Random Forest, SVM) with a Nepal focus.

**Important note on labels:** No large, openly available Nepali dataset has a ready-made "potable / not potable" label. The only dataset below with real labels is the Kaggle one. Nepali and Indian datasets contain raw measurements, so you would label them yourself against the Nepal Drinking Water Quality Standards (NDWQS). If you label by thresholds and then train on the same features, the model is only relearning your rule, so accuracy on those labels says little about real-world performance. State this clearly in your report.

**Verification note:** Links and descriptions come from web search results. Column lists are only given where the source stated them. Check each page before relying on it.

> **Document status (scope cleanup, November 2026):** This dataset survey was written during the preliminary research stage. Dataset descriptions, links, licences, and citations are preserved for reference. Some project-specific suggestions in this document (SMOTE oversampling, labeling Nepali datasets with NDWQS thresholds, combining datasets, and training on shared features across sources) are **superseded** and are not part of the approved implementation. The **finalized Purity scope** is:
> - **One** approved ML dataset: Kaggle Water Potability (`water_potability.csv`), using its nine physicochemical features and existing binary `Potability` target.
> - **Three** custom-implemented classifiers (Logistic Regression, SVM, Random Forest) built without Scikit-learn.
> - **No** synthetic training records (no SMOTE), **no** external datasets in training, **no** K-Means clustering, and **no** WQI-based categories.
> - Nepali datasets listed below serve as background context only.
>
> See the root `README.md` for the authoritative project scope.

---

## Quick Comparison

| # | Dataset | Region | Labeled? | Approx. size | Access |
|---|---------|--------|----------|--------------|--------|
| 1 | Kaggle Water Potability | Unspecified | Yes (binary) | 3,276 rows | Open (Kaggle login) |
| 2 | KUKL Kathmandu Groundwater | Kathmandu Valley, Nepal | No | 60 sources | Open (CC BY 4.0) |
| 3 | Open Data Nepal water samples | Kathmandu, Pokhara, Butwal, Birgunj, Narayanghat | No | Tiny, many files | Open (CC BY) |
| 4 | REACH-Nepal Catalyst | Western Nepal | No | 120 households | Oxford repository |
| 5 | NAST drinking water study | Nationwide, Nepal | No | 250 samples | Paper only |
| 6 | Terai arsenic testing | Terai, Nepal | No | Thousands of wells | Request from DWSS |
| 7 | CWC surface water quality | India | No | Large, 1963-2021 | Open (OGD India) |
| 8 | CPCB / GEMStat | India / global | No | Large | Open |

---

## 1. Kaggle: Water Potability (labeled)

- **Links:**
  - https://www.kaggle.com/datasets/adityakadiwal/water-potability
  - https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability (another copy of the same kind of data)
- **Description:** Water quality metrics for 3,276 different water bodies, with a binary target for human consumption. This is the standard dataset for supervised potability classification and is the best starting point for training and comparing your three models.
- **Known issues:**
  - The `ph` column has 491 missing values, and `Sulfate` and `Trihalomethanes` also have gaps. Plan an imputation strategy (median, KNN, or iterative). *(Purity decision: missing values are handled with a manually implemented imputation method whose statistics are computed from the training subset only; no Scikit-learn imputers are used.)*
  - Classes are imbalanced, with roughly 39% potable. Consider SMOTE, class weights, or stratified splits. *(Superseded: SMOTE and other synthetic oversampling are not used in Purity. The approved approach uses a manual stratified train-test split and, if justified after inspection, class weighting.)*
  - Correlations between individual features and the label are very weak (the strongest feature-feature correlation in one analysis was about 0.08). Expect modest accuracy and use cross-validation instead of one train/test split. *(Purity decision: all models are evaluated on the same reproducible, manually implemented stratified train-test split; any cross-validation, if added, must also be implemented manually.)*

### Columns

| Column | Meaning |
|--------|---------|
| `ph` | Acid-base balance of the water. WHO recommends a permissible range of 6.5 to 8.5. |
| `Hardness` | Capacity of water to precipitate soap, caused mainly by calcium and magnesium. |
| `Solids` | Total dissolved solids (TDS). High values mean highly mineralized water. |
| `Chloramines` | Chloramine concentration. Chlorine and chloramine are the main disinfectants in public water systems. |
| `Sulfate` | Sulfate concentration in the water. |
| `Conductivity` | Electrical conductivity. More dissolved ions mean higher conductivity, so it correlates with dissolved solids. |
| `Organic_carbon` | Total organic carbon (TOC). US EPA suggests under 2 mg/L in treated water and under 4 mg/L in source water. |
| `Trihalomethanes` | Chemicals that can form in chlorinated water. Levels up to 80 ppm are considered safe in drinking water. |
| `Turbidity` | Cloudiness of the water caused by suspended particles. |
| `Potability` | **Target.** 1 = safe to drink, 0 = not safe. |

---

## 2. Kathmandu Valley Groundwater (KUKL), PANGAEA

- **Link:** https://doi.pangaea.de/10.1594/PANGAEA.983648
- **Citation:** Katwal, K., Thapa, A., Pokharel, B., Ghimire, N., Dhungana, U. (2025). *Ground Water Quality Data of Kathmandu Valley for Various Source Utilized by Kathmandu Upatyaka Khanepani Limited (KUKL).* PANGAEA. https://doi.org/10.1594/PANGAEA.983648
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0). Cite the dataset when you use it.
- **Description:** Physicochemical measurements of raw groundwater from 60 operational KUKL sources, collected between September 2023 and January 2024. Arsenic and ammonia exceed NDWQS limits in urban areas, and iron and phosphate show localized contamination hotspots. All samples tested negative for E. coli, so any label you create would rest on chemistry alone.
- **Best use:** Your main Nepal case study. Label it with NDWQS thresholds. *(Superseded: KUKL data is not labeled, used as a case study, or integrated into Purity's ML pipeline; it is retained as background reference only.)*

### Columns (with reported ranges)

| Column | Meaning | Reported range |
|--------|---------|----------------|
| Coordinates | Spatial location of each source | n/a |
| Temperature | Water temperature | n/a |
| pH | Acid-base balance | n/a |
| Dissolved oxygen (DO) | Oxygen dissolved in the water | n/a |
| Ammonia | Ammonia concentration | 0.1 to 3.0 mg/L |
| Arsenic | Arsenic concentration | 0.2 to 1.5 mg/L |
| Iron | Iron concentration | 0.15 to 7.5 mg/L |
| Nitrate | Nitrate concentration | 5 to 37.5 mg/L |
| Phosphate | Phosphate concentration | 0.025 to 1.0 mg/L |
| Chloride | Chloride concentration | 0.0038 to 62.72 mg/L |
| Hardness | Total hardness | 16 to 280 mg/L |

---

## 3. Open Data Nepal: Water Sample Analyses

- **Links:**
  - All water datasets: https://opendatanepal.com/dataset?tags=water
  - Example CSV (Nakhu, Kathmandu): https://opendatanepal.com/dataset/d3955b96-9ddc-4e42-a729-bfec0460b9ac/resource/3ee17fa6-402b-45b8-b49c-be5a67c5dd90
- **License:** Creative Commons Attribution.
- **Description:** A collection of small CSV files, each analyzing water samples from three or four points along one river or lake. Sources listed include Bagmati, Bishnumati, Manohara, Hanumante, Dhobikhola, Nakhu and Taudaha (Kathmandu), Phewa Lake, Begnas Lake and Seti River (Pokhara), Tinau (Butwal), Sirsiya (Birgunj) and Narayani River (Narayanghat). Each file is tiny, but merging them gives a broader multi-city picture. These are mostly surface-water sources, not tap or well water, so treat any potability label with care.
- **Known parameters (from the Nakhu file):** pH, total dissolved solids, electrical conductivity, nitrate, total hardness and magnesium. Other files may have different columns, so harmonize them before merging.

---

## 4. REACH-Nepal Catalyst (Oxford / Eawag)

- **Link:** https://ora.ox.ac.uk/objects/uuid:7c3adef9-4cc9-45ba-a6ab-02c7f4f2310f
- **Description:** A non-randomized pre/post evaluation of five piped water supplies in Western Nepal, run by Eawag and Helvetas Nepal. The data file holds water quality results for 120 households, based on a household survey and sampling of household storage containers. The related publication is Tosi Robinson et al. (2018), *International Journal of Environmental Research and Public Health*, 15(8), 1616: https://doi.org/10.3390/ijerph15081616
- **Best use:** A real-world household-level check, and one of few sources with microbial data. Column details were not available in the search results, so open the file to see the exact variables.

---

## 5. NAST Drinking Water Study

- **Link:** https://www.nepjol.info/index.php/SW/article/view/45656
- **Description:** A study by the Nepal Academy of Science and Technology of 250 samples (135 well, 48 boring, 50 treated and 17 tap water) from across Nepal, collected December 2019 to April 2020. Values were compared with the NDWQS 2005 limits. Elevated iron, ammonia and turbidity were the most common problems.
- **Parameters assessed:** pH, conductivity, turbidity, total hardness, iron, ammonia and nitrate.
- **Access:** The page gives only summary percentages. For the raw rows, you would need to contact the authors.

---

## 6. Terai Arsenic Testing (Department of Water Supply and Sewerage)

- **Link (project background):** https://syke.fi/projects/neas
- **Description:** A blanket arsenic testing campaign ran from 2005 to 2008 across 20 Terai districts and built a national arsenic database. It is the largest real Nepali groundwater source, but it is not openly downloadable. An earlier program tested 3,896 tube wells by field kit and confirmed 1,772 samples in the lab (source: https://ircwash.org/resources/scenario-arsenic-drinking-water-nepal).
- **Access:** Request the data from DWSS. It could be valuable for an arsenic-focused label, since arsenic is the main groundwater risk in the Terai.

---

## 7. India: Central Water Commission Surface Water Quality

- **Link:** https://aikosh.indiaai.gov.in/home/datasets/details/surface_water_quality.html
- **License:** Open Government License, India.
- **Description:** Monthly surface water quality records across Indian states and union territories, spanning 1963 to 2021, collected under the Central Water Commission's monitoring programs. Reported indicators include pH, BOD and DO, among others. It is large and geographically close to Nepal, but unlabeled.

---

## 8. India CPCB and GEMStat

- **Links:**
  - CPCB database: http://cpcbedb.nic.in/
  - GEMStat India update (points to the global GEMStat database): https://gemstat.org/new-data-from-india-2023/
- **Description:** GEMStat imported Open Government Data from India's Central Water Commission and Central Pollution Control Board in 2023. GEMStat is a good place to search for data from other countries too.

---

## Suggested Workflow

> **Status:** This was the preliminary workflow proposed during dataset research. Only step 1 (and the reporting guidance in step 4) remains consistent with the finalized Purity scope. Steps 2 and 3 and the SMOTE suggestion in step 4 are **superseded**.

1. **Train and evaluate on the Kaggle dataset.** It has independent labels, so your model comparison (Logistic Regression, Random Forest, SVM) is meaningful.
2. **Apply the approach to Nepali data as a case study.** Label the KUKL data (and any Open Data Nepal files you merge) with NDWQS thresholds and state that the labels are rule-based. *(Superseded: not part of the approved scope.)*
3. **Use only shared features.** Kaggle has chloramines, trihalomethanes and organic carbon, which most Nepali datasets lack. For any combined model, train on columns both sources share (pH, hardness, TDS, conductivity, turbidity, nitrate, and so on). *(Superseded: Purity does not build combined models; all nine Kaggle features are used.)*
4. **Handle missing values and imbalance.** Use stratified cross-validation, class weights or SMOTE, and report precision, recall and F1, not accuracy alone. *(Purity decision: no SMOTE; manual stratified splitting and optional class weighting; Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix are implemented manually.)*
5. **Cite every source.** KUKL requires attribution under CC BY 4.0.
