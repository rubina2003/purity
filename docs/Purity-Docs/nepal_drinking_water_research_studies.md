# Research Studies: Drinking-Water Quality and Potability in Nepal

**Purpose:** A separate reading list of research studies relevant to the Water Potability Prediction project, with an emphasis on Nepal and Kathmandu Valley.

**Research status:** This document organizes studies identified during an initial web search. The summaries below are brief discovery notes, not a full critical review. Before citing a paper in an academic report, read the original publication and verify its bibliographic details, methods, results, and any applicable reuse licence.

> **Document status (scope cleanup, November 2026):** The study summaries and citations in this document are preserved as literature-review material. Statements about possible dataset selection or local machine-learning experiments reflect the preliminary research stage. The **finalized Purity implementation** uses only the Kaggle Water Potability dataset (`water_potability.csv`) and its existing binary `Potability` labels to train and evaluate three custom classifiers (Logistic Regression, SVM, and Random Forest). The Nepal studies listed here provide background context and are **not** integrated into model training. See the root `README.md` for the authoritative project scope.

## How these studies relate to the project

The studies can help with four distinct goals:

1. Understand drinking-water quality issues and the types of sources studied in Nepal.
2. Identify commonly measured physical, chemical, and microbiological parameters.
3. Investigate whether original sample-level data are available for analysis.
4. Compare local water-quality assessment practices with the proposed machine-learning experiment.

**Important:** A paper reporting a sample count does not necessarily provide a downloadable dataset. Do not assume its underlying sample-level measurements are available unless the paper, a repository, or the authors confirm this.

---

## 1. Potability of Bottled and Jar Water in Kathmandu Valley: A Comparative Analysis (2025)

- **Geographic focus:** Kathmandu Valley.
- **Sample count reported in the initial search:** 30 samples (15 bottled-water and 15 jar-water samples).
- **Focus:** Comparison of bottled and jar drinking water against Nepal's National Drinking Water Quality Standards (NDWQS), 2022, and WHO guidance.
- **Why it is relevant:** It closely matches the original example and provides a local approach to comparing different packaged drinking-water sources.
- **Potential limitation for machine learning:** Thirty samples are not enough to support a strong, standalone comparison of three classifiers without substantial uncertainty.
- **Data availability to investigate:** Whether sample-level measurements are supplied in the paper or can be requested from the authors.

**Links**
- [ResearchGate publication](https://www.researchgate.net/publication/393672622_Potability_of_Bottled_and_Jar_Water_in_Kathmandu_Valley_A_Comparative_Analysis)
- [Department of Environment PDF link previously identified](https://doenv.gov.np/services/Journal-of-Environment-Science-1753858957.pdf)

## 2. Microbiological Quality of Jar Water Sold in Kathmandu Valley (2009)

- **Geographic focus:** Kathmandu Valley.
- **Sample count reported in the initial search:** 30 jar-water samples representing 15 brands.
- **Focus:** Microbiological quality, including bacterial counts and tests for organisms such as *Salmonella* and *Vibrio cholerae*.
- **Why it is relevant:** It highlights that drinking-water safety includes microbiological hazards, not just chemical measurements.
- **Potential limitation:** Microbiological results may not align with the features available in a typical physicochemical water-potability dataset.
- **Data availability to investigate:** Whether results are reported per sample or only summarized by brand or test.

**Link**
- [ResearchGate publication](https://www.researchgate.net/publication/311439017_Microbial_Quality_of_Jar_Water_sold_in_Kathmandu_Valley)

## 3. Microbiological Assessment of Bottled Drinking Water of Kathmandu Valley

- **Geographic focus:** Kathmandu Valley.
- **Sample count reported in the initial search:** 30 commercial bottled-water brands/samples.
- **Focus:** Microbiological testing, including heterotrophic bacterial counts and total coliforms.
- **Why it is relevant:** It provides another local example of microbiological water-quality assessment.
- **Potential limitation:** The reported findings apply to the samples and methods in that study; they should not be generalized to all bottled water in Nepal. The exact bibliographic details and sample description should be checked in the original paper.
- **Data availability to investigate:** Whether sample-level measurements are included.

**Link**
- [ResearchGate publication](https://www.researchgate.net/publication/290455821_Microbial_assessment_of_bottled_drinking_water_of_Kathmandu_valley)

## 4. Physical, Chemical and Microbiological Characterization of Processed Drinking Water in Central Nepal: Current State Study (2020)

- **Geographic focus:** Central Nepal / Kathmandu Valley, as described in the initial search.
- **Sample count reported in the initial search:** 50 samples of processed bottled drinking water.
- **Focus:** Physical, chemical, and microbiological measurements. The initial search identified parameters including turbidity, pH, electrical conductivity, metals, fluoride, ammonia, hardness, and microbial contamination.
- **Why it is relevant:** It spans several categories of water-quality indicators and can help build a list of candidate variables to investigate.
- **Potential limitation:** A model trained on a dataset without microbiological measurements cannot establish microbiological safety.
- **Data availability to investigate:** Check the paper's tables, supplementary files, and any data-availability statement.

**Links**
- [IWA Publishing article](https://iwaponline.com/washdev/article/10/1/157/72212/Physical-chemical-and-microbiological)
- [ResearchGate publication](https://www.researchgate.net/publication/339085185_Physical_chemical_and_microbiological_characterization_of_processed_drinking_water_in_central_Nepal_current_state_study)

## 5. Groundwater Quality Evaluation for Drinking Purpose Using Water Quality Index in Kathmandu Valley, Nepal (2023)

- **Geographic focus:** Kathmandu Valley.
- **Sample count reported in the initial search:** 159 groundwater samples: 116 wells and 43 borings.
- **Methods:** Water Quality Index (WQI), correlation analysis, regression, and principal component analysis (PCA).
- **Focus:** Evaluation of groundwater quality using physicochemical and microbial measurements against Nepal's 2022 drinking-water standards.
- **Findings noted in the initial search:** The paper reported 38.36% of the 159 samples in grade E under its WQI classification. It also discussed hardness, iron, electrical conductivity, and nitrate in its multivariate analysis.
- **Why it is relevant:** It is a substantial local study and an example of using statistical methods to interpret water-quality measurements.
- **Potential limitation:** WQI grades are not automatically equivalent to the binary labels used by a machine-learning dataset. The paper's exact label definitions and data availability need to be examined.
- **Data availability to investigate:** Whether raw sample-level data or supplementary tables can be downloaded or requested.

**Link**
- [Springer Nature article](https://link.springer.com/article/10.1080/23570008.2023.2237278)

## 6. Drinking Water Quality of Kathmandu Valley (2015)

- **Geographic focus:** Kathmandu Valley.
- **Sample count reported in the initial search:** 969 samples collected between March 2012 and March 2013.
- **Source categories reported in the initial search:** 392 dug-well samples, 287 deep-boring samples, 218 treated-water samples, 46 tap-water samples, and 26 samples from other sources.
- **Focus:** Physical, chemical, and microbiological water-quality measurements, including pH, conductivity, turbidity, hardness, chloride, iron, arsenic, ammonia, nitrate, and total coliforms.
- **Why it is relevant:** Its reported sample count and range of source types make it an important lead for investigating local sample-level data.
- **Potential limitation:** The reported sample count alone does not establish that a downloadable data table exists. Sampling design, duplicate observations, and the paper's definitions need to be checked.
- **Data availability to investigate:** Tables, appendices, supplementary files, repositories, or author contact.

**Links**
- [ResearchGate publication](https://www.researchgate.net/publication/272386209_Drinking_Water_Quality_of_Kathmandu_Valley)
- [Related ResearchGate record](https://www.researchgate.net/publication/350649156_Drinking_Water_Quality_of_Kathmandu_Valley)

## 7. Ground Water Quality in the Kathmandu Valley of Nepal (2011)

- **Geographic focus:** Kathmandu Valley.
- **Focus:** Groundwater collected from shallow wells, tube wells, and deep tube wells; physical, chemical, and microbiological quality.
- **Findings noted in the initial search:** The study discussed concerns involving iron and coliform bacteria, among other parameters.
- **Why it is relevant:** It provides historical context for groundwater quality and source types.
- **Potential limitation:** Older studies may use different analytical methods, standards, units, and sampling designs from newer studies.
- **Data availability to investigate:** Whether individual sample measurements are published or can be obtained.

**Links**
- [PubMed record](https://pubmed.ncbi.nlm.nih.gov/20857191/)
- [ResearchGate publication](https://www.researchgate.net/publication/46382695_Ground_Water_Quality_in_the_Kathmandu_Valley_of_Nepal)

## 8. Drinking Water Quality in Nepal's Kathmandu Valley: A Survey and Assessment of Selected Controlling Site Characteristics

- **Geographic focus:** Kathmandu Valley.
- **Sample count:** The initial search described more than 100 water sources; verify the exact number in the original paper.
- **Source categories:** Municipal taps, dug wells, shallow and deep tube wells, and stone spouts.
- **Focus:** Water-quality assessment and relationships between contamination and water-source characteristics.
- **Why it is relevant:** It can help explain why water quality may differ by source type, beyond the measured chemistry alone.
- **Potential limitation:** Site characteristics may not be present in the proposed baseline dataset, and the original data availability is unknown.
- **Data availability to investigate:** Exact sample count, sample-level tables, and data-access options.

**Link**
- [ResearchGate publication](https://www.researchgate.net/publication/226478961_Drinking_water_quality_in_Nepal%27s_Kathmandu_Valley_A_survey_and_assessment_of_selected_controlling_site_characteristics)

## 9. Drinking Water Quality and Public Health in the Kathmandu Valley, Nepal: Coliform Bacteria, Chemical Contaminants, and Health Status of Consumers (2022)

- **Geographic focus:** Kathmandu Valley.
- **Sample count reported in the initial search:** 35 water sources.
- **Source categories:** Tube wells, dug wells, stone spouts, and municipal taps.
- **Focus:** Bacterial and chemical contaminants, including coliforms, arsenic, fluoride, iron, lead, mercury, and other metals; the study also collected information about households using sampled sources.
- **Why it is relevant:** It connects measured water quality with broader public-health considerations.
- **Potential limitation:** Association with health status does not by itself establish causation, and its sample count may be too small for a broad machine-learning model.
- **Data availability to investigate:** Whether sample-level measurements and household-level data are available, and under what conditions.

**Link**
- [Wiley / Journal of Environmental and Public Health article](https://onlinelibrary.wiley.com/doi/10.1155/2022/3895859)

## 10. Quality of Drinking Water in Kathmandu Valley, Nepal (2019)

- **Geographic focus:** Kathmandu Valley.
- **Sample count reported in the initial search:** 60 samples: 20 each from public supplies, private supplies, and processed drinking-water sources.
- **Focus:** Mineral measurements and bacterial contamination across different supply types.
- **Why it is relevant:** It offers a source-category comparison and broadens the review beyond groundwater.
- **Potential limitation:** Confirm the exact methods, sample units, and target definitions in the original paper before comparing it with other studies.
- **Data availability to investigate:** Whether the original sample-level measurements are reported or obtainable.

**Link**
- [ResearchGate publication](https://www.researchgate.net/publication/335385707_Quality_of_drinking_water_in_Kathmandu_valley_Nepal)

## 11. Emerging Threats on Drinking Water Safety in Kathmandu: A Study of Physicochemical and Bacterial Parameters (2025)

- **Geographic focus:** Kathmandu Valley.
- **Sample count reported in the initial search:** 30 samples: 15 tap-water and 15 jar-water samples.
- **Collection period reported in the initial search:** November 2023 to April 2024.
- **Focus:** Physicochemical and bacterial parameters.
- **Why it is relevant:** It is a relatively recent local study that may help identify current research questions.
- **Potential limitation:** The small sample size limits the strength of any standalone machine-learning experiment.
- **Data availability to investigate:** Verify the paper's publication details, methods, results, and whether sample-level data are available.

**Link**
- [ResearchGate publication](https://www.researchgate.net/publication/404169612_Emerging_Threats_on_Drinking_Water_Safety_in_Kathmandu_A_Study_of_Physicochemical_and_Bacterial_Parameters)

## 12. Seasonal Differences in Enteric Pathogen Occurrence Across Multiple Drinking Water Sources in Kathmandu Valley, Nepal (2026)

- **Geographic focus:** Kathmandu Valley.
- **Sample count and period reported in the initial search:** 86 samples collected in 2018; 22 in winter and 64 in summer.
- **Source categories:** Tanker, jar, tap, well, and stone-spout water.
- **Focus:** Enteric pathogen occurrence and seasonal differences.
- **Why it is relevant:** It highlights season and water-source type as potential influences on observed water quality.
- **Potential limitation:** Pathogen detection data may not be directly compatible with a physicochemical baseline dataset.
- **Data availability to investigate:** The article's data statement, supplementary files, and access conditions.

**Link**
- [IWA Publishing / DOI record](https://doi.org/10.2166/wh.2026.238)

---

# Cross-study comparison

The following is a discovery-stage comparison based on the initial search notes. It is not a claim that all studies have comparable samples or measurements.

| Study | Main water source(s) | Reported sample count | Primary relevance |
|---|---|---:|---|
| Bottled and Jar Water Comparative Analysis (2025) | Bottled, jar | 30 | Packaged-water comparison |
| Microbiological Quality of Jar Water (2009) | Jar | 30 | Microbiological quality |
| Microbiological Assessment of Bottled Water | Bottled | 30 | Microbiological quality |
| Processed Drinking Water in Central Nepal (2020) | Processed bottled | 50 | Broad physicochemical and microbial assessment |
| Groundwater Quality Evaluation (2023) | Wells, borings | 159 | WQI and multivariate analysis |
| Drinking Water Quality of Kathmandu Valley (2015) | Wells, borings, treated, tap, other | 969 | Broad local survey |
| Ground Water Quality in Kathmandu Valley (2011) | Shallow and deep groundwater | Verify in paper | Historical groundwater context |
| Kathmandu Valley source-characteristics survey | Multiple sources | More than 100; verify | Source-related contamination patterns |
| Drinking Water Quality and Public Health (2022) | Multiple sources | 35 | Contaminants and consumer context |
| Quality of Drinking Water in Kathmandu Valley (2019) | Public, private, processed | 60 | Supply-type comparison |
| Emerging Threats (2025) | Tap, jar | 30 | Recent physicochemical and bacterial assessment |
| Seasonal Pathogen Occurrence (2026) | Multiple sources | 86 | Seasonal pathogen differences |

**Do not combine these sample counts or datasets.** Studies may use different sampling designs, measurement methods, units, definitions, and overlapping sources.

---

# What to extract when reading each paper

Use this checklist to build a consistent literature-review spreadsheet or notes file.

- [ ] Full title, authors, year, journal, DOI, and stable URL.
- [ ] Study location and sampling dates.
- [ ] Water-source categories.
- [ ] Number of samples and how samples were selected.
- [ ] Parameters measured and their units.
- [ ] Laboratory methods and detection limits, where reported.
- [ ] Standard or guideline used, including its edition.
- [ ] How the authors defined acceptable, unsafe, or potable water.
- [ ] Main findings and the authors' stated limitations.
- [ ] Whether sample-level data are available.
- [ ] Data licence or permission requirements.
- [ ] Whether the paper is suitable for background research, dataset construction, external validation, or only contextual comparison.

## Key cautions for the machine-learning project

1. **Potability labels are not automatically equivalent across studies.** A study's WQI grade, a standard-based pass/fail result, and a laboratory microbiological result are different targets.
2. **Do not pool measurements until compatibility is established.** Check units, methods, feature definitions, dates, and source populations.
3. **Avoid circular evaluation.** If labels are generated from threshold rules applied to the same features given to the model, explain this clearly and do not present the result as independent health-risk prediction.
4. **Chemical measurements do not establish microbial safety.** State which hazards are and are not represented by the model.
5. **Do not assume the reported results generalize to all of Nepal.** Most studies here focus on specific samples and locations in Kathmandu Valley.
6. **Verify every citation before using it academically.** The links and findings in this initial list are starting points; the original papers should be treated as the authoritative source.

## Suggested next steps

> **Status:** These were preliminary research steps. Steps related to reading and verifying the literature remain useful for the academic report. Steps concerning the selection or construction of a local training dataset (items 2–4 and 6) are **superseded**: the baseline dataset decision has been finalized as the Kaggle Water Potability dataset.

1. Read the 2023 groundwater study and the 2015 Kathmandu Valley survey first.
2. Check their full text and supplementary material for sample-level data.
3. Investigate the 2020 processed-water study for its measured parameters and data-availability statement.
4. Build a data-availability table recording `available`, `request required`, `summary only`, or `unknown`.
5. Separately search for studies that implement machine-learning models for water-potability prediction; the studies in this list primarily help characterize the local water-quality problem and should not all be treated as machine-learning papers.
6. Keep the baseline dataset decision open until the audit and data-discovery work is complete. *(Superseded: the baseline dataset has been finalized as `water_potability.csv` from Kaggle.)*
