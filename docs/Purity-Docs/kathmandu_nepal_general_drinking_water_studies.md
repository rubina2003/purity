# Drinking-Water Quality Research in Kathmandu Valley and Nepal
## Broader literature map (not limited to bottled or jar water)

**Purpose:** This document focuses on drinking water as a whole: groundwater, surface water, municipal/tap supplies, springs, stone spouts (dhunge dharas), household drinking water, and national rural/urban water-quality patterns. Packaged-water studies are included only when they add useful context.

**Research scope:** Kathmandu Valley is a priority because it is the project's local context, but the list also includes Nepal-wide evidence. The document is intended for research discovery and literature review, not as a substitute for reading the original papers.

**Important distinction:** Most of these are water-quality assessment studies, not machine-learning papers. They can explain local water-quality challenges and help identify candidate measurements or datasets. They do not automatically provide downloadable sample-level data.

> **Document status (scope cleanup, November 2026):** This literature map was compiled during the preliminary research stage. The study summaries, findings, and citations below are preserved as academic background. Where this document discusses possible project decisions (for example, using local datasets for model training or validation), those were preliminary suggestions. The **finalized Purity implementation** trains and evaluates its three custom classifiers (Logistic Regression, SVM, and Random Forest) **only** on the Kaggle Water Potability dataset (`water_potability.csv`) with its existing binary `Potability` labels. Local studies and datasets are used solely as background context. WQI methods described in individual studies are reported as research findings only — WQI-based categories are not part of Purity's scope. See the root `README.md` for the authoritative project scope.

---

## Highest-priority papers to start with

### 1. Microbial and Chemical Water Quality Assessments Across the Rural and Urban Areas of Nepal: A Scoping Review (2025)

- **Scope:** Nationwide review of research on drinking and irrigation water in rural and urban Nepal.
- **Evidence base:** 140 studies met the review's inclusion criteria.
- **Main value:** This is the best starting point for broadening the literature search. It maps the types of water sources, locations, and contaminants investigated across Nepal.
- **Key findings reported by the review:** Bacteria and arsenic were among the most frequently reported contaminants. The authors identify uneven geographic coverage, inconsistent testing protocols, weak regulatory enforcement, and limited integration of water-quality monitoring with public-health planning as important gaps.
- **Why read it first:** Its reference list and supplementary list of included studies can lead to many additional papers, including studies beyond Kathmandu and beyond bottled/jar water.
- **Data potential:** It is a review, not a single raw dataset. Use its included-study list to locate primary studies and investigate their data availability.

**Links**
- [Full open-access article (MDPI)](https://www.mdpi.com/1660-4601/22/10/1526)
- [PubMed record](https://pubmed.ncbi.nlm.nih.gov/41154930/)
- [University of Maryland repository, with supplementary files](https://drum.lib.umd.edu/items/bc6fb9d4-0904-4da2-97fb-411c92fa1002)

### 2. Extent of Fecal Contamination of Household Drinking Water in Nepal: Further Analysis of Nepal Multiple Indicator Cluster Survey 2014

- **Scope:** Nepal-wide household drinking-water evidence using the 2014 Nepal Multiple Indicator Cluster Survey (MICS).
- **Main value:** More nationally relevant than a small, single-city source study.
- **Finding reported by the paper:** Fecal contamination was detected in 81.2% of household drinking water from improved sources and 89.6% from unimproved sources; the adjusted analysis found no difference in odds of contamination between the two categories.
- **Why it matters:** It demonstrates that a source classified as “improved” does not necessarily guarantee safe water at the point of household use.
- **Data potential:** Investigate the original MICS data and documentation, including whether the relevant water-quality variables are accessible and how samples/households are represented. Do not assume the paper's analysis file is a ready-to-use ML dataset.

**Link**
- [Monash University research record](https://research.monash.edu/en/publications/extent-of-fecal-contamination-of-household-drinking-water-in-nepa/)

### 3. Drinking Water Quality in Nepal's Kathmandu Valley: A Survey and Assessment of Selected Controlling Site Characteristics (2008; online publication dates may differ)

- **Scope:** Multiple drinking-water source types in Kathmandu Valley.
- **Sources:** Municipal taps, dug wells, shallow- and deep-aquifer tube wells, and stone spouts (dhunge dharas).
- **Sample count:** More than 100 sources, according to the abstract.
- **Finding reported:** Total coliform and *E. coli* were detected in 94% and 72% of sampled sources, respectively. The study also investigated nitrate, ammonia, and metals.
- **Why it matters:** It directly compares source types and explores how source/site characteristics relate to contamination.
- **Data potential:** The full paper is accessible through some repositories. Check its tables and supplements for sample-level data.

**Links**
- [Penn State research record](https://pure.psu.edu/en/publications/drinking-water-quality-in-nepals-kathmandu-valley-a-survey-and-as/)
- [ResearchGate record / full-text availability](https://www.researchgate.net/publication/226478961_Drinking_water_quality_in_Nepal%27s_Kathmandu_Valley_A_survey_and_assessment_of_selected_controlling_site_characteristics)
- [Author-hosted PDF](https://bohs.biology.utah.edu/PDFs/Frank/Warner%20etal%202007.pdf)

### 4. Drinking Water Quality of Kathmandu Valley (2015)

- **Scope:** Broad survey of drinking-water sources in Kathmandu Valley.
- **Sample count reported in the initial search:** 969 samples collected from March 2012 to March 2013.
- **Sources reported:** Dug wells, deep borings, treated water, taps, and other sources.
- **Parameters reported in the initial search:** pH, conductivity, turbidity, hardness, chloride, iron, arsenic, ammonia, nitrate, and total coliforms.
- **Why it matters:** The reported sample count and diversity of sources make it a high-priority lead for understanding local water quality at a broader scale.
- **Data potential:** Verify the original paper's exact sampling design and whether the 969 observations are available in a table or downloadable dataset. The sample count alone does not mean raw data are public.

**Links**
- [ResearchGate publication](https://www.researchgate.net/publication/272386209_Drinking_Water_Quality_of_Kathmandu_Valley)
- [Related ResearchGate record](https://www.researchgate.net/publication/350649156_Drinking_Water_Quality_of_Kathmandu_Valley)

---

## Kathmandu Valley: multi-source and groundwater studies

### 5. Groundwater Quality Evaluation for Drinking Purpose Using Water Quality Index in Kathmandu Valley, Nepal (2023)

- **Sources/sample count:** 159 groundwater samples: 116 wells and 43 borings.
- **Methods:** Water Quality Index (WQI), correlation analysis, regression, and principal component analysis (PCA).
- **Scope:** Physicochemical and microbial measurements assessed against Nepal's 2022 drinking-water standards.
- **Finding reported:** The paper found that several parameters, including conductivity, turbidity, iron, ammonia, total coliform, and *E. coli*, exceeded the national standard in some portion of samples. It also compared well water and boring water.
- **Why it matters:** A substantial local groundwater study with multivariate analysis and an explicit standards-based assessment.
- **Data potential:** Check supplementary files or contact the authors for individual sample measurements.
- **Caution:** WQI classes are not automatically interchangeable with binary potable/non-potable labels.

**Links**
- [Springer Nature article](https://link.springer.com/article/10.1080/23570008.2023.2237278)
- [Full-text version at Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/23570008.2023.2237278)

### 6. Water Quality Status of Groundwater and Municipal Water Supply (Tap Water) from Bagmati River Basin in Kathmandu Valley, Nepal (2021 journal issue; DOI registered in 2020)

- **Sources/sample count:** 52 samples from deep tube wells, tube wells, dug wells, and tap water.
- **Focus:** Physical, chemical, and microbiological quality.
- **Finding reported:** Total coliform exceeded drinking-water guidance in 84.6% of samples. The study also detected several enteric bacteria.
- **Why it matters:** It compares groundwater and municipal tap supplies within the same study rather than focusing on one packaged-water category.
- **Data potential:** The paper is open access under CC BY 4.0 according to the record; check the publisher's article and any supplementary data before reuse.

**Links**
- [ResearchGate full-text record](https://www.researchgate.net/publication/347922667_Water_quality_status_of_groundwater_and_municipal_water_supply_tap_water_from_Bagmati_river_basin_in_Kathmandu_valley_Nepal)
- [IWA Publishing DOI page](https://doi.org/10.2166/washdev.2020.190)

### 7. Drinking Water Quality and Public Health in the Kathmandu Valley, Nepal: Coliform Bacteria, Chemical Contaminants, and Health Status of Consumers (2022)

- **Sources/sample count:** 35 tube wells, dug wells, stone spouts, and municipal tap-water sources.
- **Additional context:** Household survey covering 146 households and 603 individuals.
- **Focus:** Total/fecal coliforms and a broad range of chemical contaminants, including metals.
- **Finding reported:** 94% of sampled sources had detectable total or fecal coliform. Some samples exceeded standards or health-based values for aluminum, arsenic, iron, and manganese.
- **Why it matters:** Connects multi-source water testing with consumer context and public-health considerations.
- **Data potential:** Check the article's data-availability statement and supplementary material.

**Links**
- [Open-access full text (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8858048/)
- [Wiley article](https://onlinelibrary.wiley.com/doi/10.1155/2022/3895859)
- [PubMed record](https://pubmed.ncbi.nlm.nih.gov/35190745/)

### 8. Ground Water Quality in the Kathmandu Valley of Nepal (2011)

- **Sources:** Shallow wells, tube wells, and deep tube wells across Kathmandu, Lalitpur, and Bhaktapur.
- **Focus:** Physical, chemical, and microbiological groundwater quality.
- **Finding reported:** The study identified concerns involving iron and coliform bacteria; it also measured conductivity, turbidity, and other parameters.
- **Why it matters:** Adds historical groundwater context and shows how risks differ across groundwater sources.
- **Data potential:** Verify whether individual measurements are published or obtainable.

**Links**
- [PubMed record](https://pubmed.ncbi.nlm.nih.gov/20857191/)
- [DOI](https://doi.org/10.1007/s10661-010-1706-y)

### 9. Assessment of Drinking Water Quality of Kathmandu Municipality Area, Kathmandu, Nepal in Pre-Monsoon Season (2019)

- **Scope:** Kathmandu Municipality area.
- **Sample count:** 97 samples.
- **Sources:** Stone spouts, wells, borings, and municipal public taps.
- **Parameters:** Turbidity, electrical conductivity, pH, hardness, sulfate, chloride, nitrate, ammonia, iron, manganese, arsenic, and total coliform.
- **Why it matters:** A multi-source local assessment with chemical and microbial variables.
- **Data potential:** Review the article's tables for sample-level versus aggregate results.

**Link**
- [Research Journal of Recent Sciences article](https://www.isca.me/rjrs/archive/v8/i4/7.ISCA-RJRS-2019-029.php)

### 10. Evaluation of Surface Water and Groundwater Quality for Drinking Purposes Using the Water Quality Index: A Case Study of Kathmandu Valley, Nepal (2026)

- **Sources/sample count:** 76 public water-supply sources: 22 surface-water and 54 groundwater sources.
- **Collection period:** May–July 2024.
- **Parameters:** Physical properties including turbidity, pH, colour, taste/odour and electrical conductivity, plus chemical parameters such as iron, manganese, arsenic, fluoride, ammonia, chloride, sulphate, nitrate, copper, zinc, aluminum, and total hardness.
- **Finding reported:** The study found differences in standards compliance between surface-water and groundwater samples and used WQI and correlation analysis.
- **Why it matters:** Directly broadens the scope to both surface water and groundwater used for public supply.
- **Data potential:** The publisher page should be checked for the data statement and supplementary material.

**Links**
- [IWA Publishing article](https://iwaponline.com/washdev/article-abstract/16/3/176/111145/Evaluation-of-surface-water-and-groundwater)
- [ResearchGate record](https://www.researchgate.net/publication/402052435_Evaluation_of_surface_water_and_groundwater_quality_for_drinking_purposes_using_the_water_quality_index_a_case_study_of_Kathmandu_Valley_Nepal)

### 11. Seasonal Dynamics of Spring Water Quality and Quantity in Urban Cities of Kathmandu Valley, Central Nepal (2026)

- **Sources/sample count:** 150 springs mapped; water-quality measurements reported for 146 springs post-monsoon and 142 pre-monsoon.
- **Collection periods:** November 2023 and March 2024.
- **Focus:** Spring water quality and quantity, seasonal changes, and suitability for drinking, irrigation, and aquatic life.
- **Why it matters:** Adds a substantial spring-water study and explicitly investigates seasonal variation.
- **Finding reported:** The study found seasonal differences in spring discharge and WQI classifications. It also reported coliform detection in a subset of spring sources.
- **Data potential:** Open access; inspect supplementary tables and the article's data availability.
- **Caution:** The paper reports both drinking-water and aquatic-life comparisons; keep those standards and endpoints distinct.

**Link**
- [Springer Nature open-access article](https://link.springer.com/article/10.1007/s12665-026-12915-3)

### 12. Seasonal Differences in Enteric Pathogen Occurrence Across Multiple Drinking Water Sources in Kathmandu Valley, Nepal (2026)

- **Sample count:** 86 samples collected in 2018.
- **Sources:** Tanker, jar, tap, well, and stone-spout water.
- **Focus:** Molecular screening for enteric pathogens and seasonal differences.
- **Why it matters:** Illustrates that source type and season can affect microbial findings.
- **Data potential:** Review the publisher's data statement and supplementary material.
- **Caution:** Pathogen-detection data are not automatically compatible with a chemical-feature potability dataset.

**Link**
- [IWA Publishing / DOI page](https://doi.org/10.2166/wh.2026.238)

### 13. Status of Public Drinking Water Quality of Multiple Sources in Kathmandu Valley, Nepal (2026)

- **Sample count:** 185 samples collected during 2024–2025.
- **Sources:** Tap water (68), jar water (16), bottled water (21), and household-processed water (80).
- **Focus:** Integrated physicochemical and microbiological testing against NDWQS.
- **Why it is included despite packaged-water categories:** The study compares several common household drinking-water sources, not only bottled or jar water.
- **Finding reported:** The paper identifies microbial contamination as a key concern and cautions that a physicochemical WQI does not establish microbiological safety.
- **Data potential:** The publisher states that data are available on request; follow the paper's data-availability instructions.
- **Publication timing:** Search results identify it as an October 2026 article. Verify the final publication metadata and article status before citing it.

**Link**
- [ScienceDirect article](https://www.sciencedirect.com/science/article/pii/S1944398626002250)

### 14. Traditional Stone Water Spouts Status and Its Practical Significance in Urbanizing Kathmandu Valley, Nepal – A Review (2022)

- **Type:** Review paper.
- **Focus:** The status of traditional stone spouts, their water quantity and quality, and the effects of urbanization, groundwater extraction, sewage, and other contamination risks.
- **Why it matters:** Provides broader context about an important traditional source used by communities, rather than only testing individual bottled products.
- **Data potential:** Use its references to find primary studies of individual spouts and source conditions.

**Link**
- [ScienceDirect open-access review](https://www.sciencedirect.com/science/article/pii/S2667010022001299)

### 15. Stone Spout Water Quality for Drinking Purposes in Kathmandu Metropolitan: An Assessment Using Water Quality Index (2025)

- **Scope:** Stone spouts in Kathmandu Metropolitan.
- **Focus:** Assessment of traditional public drinking-water sources using WQI.
- **Why it matters:** Relevant to local public-source drinking water and can complement broader groundwater and municipal-supply studies.
- **Data potential:** Read the paper for sample count, sampling dates, exact parameters, standards, and whether raw data are available.

**Link**
- [NepJOL article PDF](https://www.nepjol.info/index.php/JNCS/article/view/74490/57107)

---

## Broader Nepal context: use alongside Kathmandu-specific papers

### 16. Nepal Multiple Indicator Cluster Survey (MICS) 2014 — source data and survey documentation

- **Type:** National household survey, not a single journal article.
- **Why it matters:** The related analysis of fecal contamination uses this survey to compare household drinking water from improved and unimproved sources. Survey documentation is essential for understanding sample design, household representation, variables, and limitations.
- **Data potential:** Investigate access to original survey microdata and water-quality variables through the official survey/data channels. Do not treat an article's percentages as a sample-level training dataset.

**Starting point**
- [Kandel et al. analysis of MICS 2014](https://research.monash.edu/en/publications/extent-of-fecal-contamination-of-household-drinking-water-in-nepa/)

### 17. National and regional arsenic / groundwater studies

For a Nepal-wide picture, search the references in the 2025 scoping review for studies on:
- Arsenic in Terai groundwater and tube wells.
- Rural and urban groundwater contamination.
- Microbial contamination in household drinking water.
- Spring water quality in hill and mountain districts.
- Water-quality monitoring in multiple provinces.

**Why this is a research direction rather than a single citation:** Nepal's water-quality literature is geographically diverse, and arsenic studies may use different sample designs, units, and endpoints. Use the scoping review's included-study list to select and verify individual primary papers instead of treating one regional study as representative of the whole country.

---

# Comparison table

Sample counts below are those reported in the cited paper or in the initial search notes. Check each original article before relying on the number in academic work.

| Study | Geographic scope | Main sources | Reported sample size | Main contribution |
|---|---|---|---:|---|
| Nepal rural/urban scoping review (2025) | Nationwide literature | Many types | 140 included studies | Best map of broader Nepal literature |
| MICS 2014 fecal-contamination analysis | Nepal-wide household survey | Improved/unimproved household sources | See original analysis | National household-level context |
| Warner et al. source-characteristics survey (2008) | Kathmandu Valley | Tap, wells, tube wells, stone spouts | 100+ sources | Source type and contamination patterns |
| Kathmandu Valley water-quality survey (2015) | Kathmandu Valley | Wells, borings, treated, taps, other | 969 reported | Large local multi-source survey |
| Groundwater WQI study (2023) | Kathmandu Valley | Wells and borings | 159 | Groundwater, WQI, multivariate statistics |
| Groundwater and municipal tap study (2021 issue) | Bagmati basin / Kathmandu Valley | Groundwater and tap | 52 | Chemical and microbial comparison |
| Public health and water-quality study (2022) | Kathmandu Valley | Wells, spouts, taps | 35 sources | Contaminants plus household context |
| Groundwater study (2011) | Kathmandu Valley | Shallow/deep wells | Verify in paper | Historical groundwater assessment |
| Kathmandu Municipality pre-monsoon study (2019) | Kathmandu Municipality | Spouts, wells, borings, taps | 97 | Multi-source physicochemical and microbial data |
| Surface water and groundwater WQI study (2026) | Kathmandu Valley | Surface and groundwater | 76 | Public-supply sources and WQI |
| Spring quality/quantity study (2026) | Kathmandu Valley | Springs | 150 mapped | Seasonal water quality and discharge |
| Enteric pathogen seasonal study (2026) | Kathmandu Valley | Tanker, jar, tap, wells, spouts | 86 | Microbial pathogens and seasonality |
| Public drinking water quality study (2026) | Kathmandu Valley | Tap, jar, bottled, processed | 185 | Multiple household source types |
| Stone-spout review (2022) | Kathmandu Valley | Stone spouts | Review | Source conservation and contamination context |
| Stone-spout WQI study (2025) | Kathmandu Metropolitan | Stone spouts | Verify in paper | Traditional public source quality |

**Do not add these sample counts together.** Studies may overlap in location or sampling period, measure different things, and use incompatible methods or labels.

---

# What this means for the Water Potability Prediction project

> **Finalized decision:** The baseline dataset question raised below has since been resolved. Purity uses the Kaggle Water Potability dataset as its single approved ML training and evaluation dataset. The local studies listed in this document inform the problem background, the limitations discussion, and the separate parameter-level standards assessment; they are **not** integrated into model training. The recommendations below are retained as the original research reasoning.

The broader research should inform the problem definition, but it should not force a change to the baseline dataset before the evidence is assessed.

## Keep three questions separate

1. **What are the drinking-water quality problems in Nepal?**  
   Answer this with nationwide reviews, public-health studies, and local multi-source assessments.

2. **What sample-level data can we actually access and reuse?**  
   Investigate raw data, survey microdata, supplementary tables, author requests, units, sample metadata, and licences.

3. **What can the proposed classifier validly predict?**  
   This depends on the features and target labels in the chosen dataset, not just on what local water-quality studies measure.

## Key limitations to document

- A study's WQI category is not automatically the same as a potable/non-potable binary label.
- Microbial and chemical quality are different dimensions of safety. A dataset lacking microbial variables cannot establish microbial safety.
- Labels derived from thresholds applied to the same input features can make a classifier's task circular; report this explicitly.
- Samples from different regions, seasons, sources, or laboratory methods should not be pooled until comparability is demonstrated.
- A Kathmandu Valley study is not automatically representative of all Nepal.
- A large published sample count does not prove that individual measurements are publicly downloadable.
- Do not use a model trained on an unrelated dataset as ground truth for Nepalese samples.

## Recommended research workflow

1. Read the **2025 Nepal-wide scoping review** first and use its included-study supplement as a reference map.
2. Read the **MICS 2014 fecal-contamination analysis** for nationwide household drinking-water context.
3. Read the **Kathmandu Valley multi-source survey**, **2015 broader survey**, and **2023 groundwater WQI study** for local source and parameter detail.
4. Create a separate data-availability table with status values such as `public raw data`, `supplementary table`, `request required`, `summary only`, or `unknown`.
5. For each possible dataset, record source type, geography, date, sample unit, variables and units, lab method, target definition, missingness, licence, and data-access status.
6. Only after this audit decide whether a local dataset can support model training, external validation, or background context only. *(Superseded: the finalized Purity scope uses local datasets as background context only; they are not used for model training or external validation.)*
7. Separately search for machine-learning water-potability studies. Most papers in this list establish the water-quality context rather than evaluate Logistic Regression, Random Forest, or SVM from scratch.
