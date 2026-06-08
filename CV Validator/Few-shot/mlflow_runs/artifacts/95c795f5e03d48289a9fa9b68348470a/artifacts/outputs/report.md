# AI CV validator

**Pozicia:** AI Engineer
**Seniorita:** unknown
**Zdroj poziadaviek:** llm_json_plus_fallback_merge
**Celkove skore:** 55.51 / 100
**Odporucanie:** Ciastocne vhodny / vyzaduje manualne posudenie

> Vystup je odporucanie pre cloveka, nie automaticke rozhodnutie o kandidatovi.

## Anonymizovany profil kandidata
Nepodarilo sa spolahlivo extrahovat profil.

## Vyhodnotenie poziadaviek

| Stav | Poziadavka | Skore | Priorita | Vysvetlenie |
|---|---|---:|---|---|
| ❌ nesplnene | University education (Bachelor's degree, Master's degree, or Doctorate) | 0 | must / w=5.0 | CV uvadi len vyskolu na vysokej skole a certifikacie, vyskolu na vysokej skole nie je uvadeny. |
| ✅ splnene | Degree in Computer Science, Applied Mathematics, Software Engineering, or equivalent work experience | 85 | must / w=5.0 | Kandidat nemá uvedený študijný titul v oblasti počítačových vied, ale má 15 rokov praxe a rozsiahle skúsenosti s vývojom, automatizáciou a architektúrou, čo splňuje kritérium ekvivalentnej pracovnej skúsenosti. |
| ✅ splnene | Proficiency in Python or similar languages | 92 | must / w=5.0 | Kandidat explicitne uvádza Python v zozname jadroch a opisuje skusenosti s jeho pouzivanim v praxi. |
| ✅ splnene | Experience with modern AI/ML frameworks | 85 | must / w=4.0 | Kandidat explicitne uvodi skusenost s modernymi AI/ML frameworkmi ako XGBoost a Scikit-learn, ako aj vyvoj prediktivnych modelov a patentov v oblasti AI/ML. |
| 🟡 ciastocne_splnene | Experience with Generative AI | 65 | must / w=4.0 | Kandidat ma skusenost s AI a ML modelmi (prediktivna analytika, XGBoost), ale explicitne sa nezmieta generativna AI. |
| ❌ nesplnene | Experience with Natural Language Processing | 0 | must / w=4.0 | CV uvadi skusenosti s ML modelmi (XGBoost) a spracovanym datom, ale bez explicitneho ziadania NLP. |
| ❌ nesplnene | English language at Upper intermediate (B2) level | 0 | must / w=3.0 | V dodanych odkazoch z CV nie je zistena informacia o anglictine. |
| ✅ splnene | Experience deploying AI solutions including model monitoring and maintenance | 85 | nice / w=2.0 | Kandidat vyvoj modelov a ich monitorovanie (Grafana, alerting) je v CV jasne uvedeny. |
| ✅ splnene | Experience working with large-scale data pipelines and distributed computing | 85 | nice / w=2.0 | Kandidat explicitne uvodzuje skusenost s budovanim a udrzavanim datovych rurovodov (data pipelines) pre monitorovanie a analytiku, ako aj s automatizaciou a CI/CD. |
| ✅ splnene | Solid understanding of machine learning, deep learning, and MLOps methodologies | 85 | nice / w=2.0 | Kandidat explicitne uvodi skusenosti s Machine Learningom (XGBoost, Scikit-learn, AutoML) a aplikaciu v projektoch na predikciu anomaliou a vytvoreni patentov. |
| ✅ splnene | Ability to explain technical concepts to non-technical audience | 85 | nice / w=2.0 | Kandidat vykazuje skusenost s vedenim internych technickych sedení, mentorstvom a trenim novych inzhinierov, co naznacuje schopnost vysvetlovat technicke pojmy. |
| ❌ nesplnene | Enthusiasm for continuous learning and trying new approaches | 0 | nice / w=1.0 | V dodanych odkazoch z CV nie je zistena ziskna skusenost s ucenim sa. |

## Odkazy a poznamky

### ❌ R1: University education (Bachelor's degree, Master's degree, or Doctorate)
**Riziko/neistota:** Chyba v chudzke vyskolu na vysokej skole.
**Pouzite odkazy:**
- [similarity=0.270] Faculty of management Electrotechnical High School, Slovakia (2001–2005) • ITIL v3 Foundation (2015) • Cisco Certified Design Associate (CDA) and Cisco Certified Design Professional (CDP) • Cisco Certified Network Associate (CNA) Recent Key Projects & Achievements • AI-driven Infrastructure Anomaly Detection: Developed a Python + SQL + Grafana solution predicting call volume anomalies in real-time, which reduced false positives • Patent Activity: Filed multiple patents in AI/ML applied to IT infrastructure and automation analytics. • Cross-domain Mentorship: Led internal technical sessions on automation, telemetry visualization, and Python data engineering for IT teams.

### ✅ R2: Degree in Computer Science, Applied Mathematics, Software Engineering, or equivalent work experience
**Riziko/neistota:** Chýbajúci formálny titul v požadovanej oblasti, ale silné praktické skúsenosti.
**Pouzite odkazy:**
- Faculty of management Electrotechnical High School, Slovakia (2001–2005)
- 15+ years of international experience in data-driven infrastructure design, automation, and analytics
- Core Skills: Python, Pandas, Numpy, SQL, Polars, Power BI, Grafana, Kibana

### ✅ R3: Proficiency in Python or similar languages
**Pouzite odkazy:**
- Core Skills • Programming & Data: Python, Pandas, Numpy, SQL, Polars, Power BI, Grafana, Kibana
- Engineered and maintaining data extraction, processing, and visualization pipelines for network monitoring and performance analytics (MS SQL, Python, Logstash, Grafana)
- Developed ML-based predictive models (XGBoost) to identify anomalies in infrastructure telemetry

### ✅ R4: Experience with modern AI/ML frameworks
**Riziko/neistota:** Skusenost je priama, ale nie je uvedena detailna zoznamovacia skusenost s inymi frameworkmi ako PyTorch alebo TensorFlow.
**Pouzite odkazy:**
- Developed ML-based predictive models (XGBoost)
- Machine Learning: AutoML, XGBoost, Scikit-learn, Predictive Analytics
- AI-driven Infrastructure Anomaly Detection: Developed a Python + SQL + Grafana solution predicting call volume anomalies in real-time

### 🟡 R5: Experience with Generative AI
**Riziko/neistota:** Skusenost je obmedzena na tradičné ML a prediktívne modely, nie na generatívne technológie.
**Pouzite odkazy:**
- AI-driven Infrastructure Anomaly Detection: Developed a Python + SQL + Grafana solution predicting call volume anomalies in real-time
- Patent Activity: Filed multiple patents in AI/ML applied to IT infrastructure and automation analytics
- ML-based predictive models (XGBoost) to identify anomalies in infrastructure telemetry

### ❌ R6: Experience with Natural Language Processing
**Riziko/neistota:** Chyba explicitneho ziadania NLP v CV.

### ❌ R7: English language at Upper intermediate (B2) level
**Riziko/neistota:** Poziadavka sa nepodarilo podlozit ziadnym dokazom z CV.

### ✅ R8: Experience deploying AI solutions including model monitoring and maintenance
**Riziko/neistota:** Bez priameho uvodzenia slova deploy, ale skusenost s CI/CD a monitoringom je silny indikator.
**Pouzite odkazy:**
- Developed ML-based predictive models (XGBoost) to identify anomalies in infrastructure telemetry, integrated with CI/CD pipelines in GitLab.
- AI-driven Infrastructure Anomaly Detection: Developed a Python + SQL + Grafana solution predicting call volume anomalies in real-time
- Lifecycle management of monitoring and alerting applications within Telecom environment (IR Prognosis, Oracle EOM)

### ✅ R9: Experience working with large-scale data pipelines and distributed computing
**Riziko/neistota:** Termin distributed computing nie je priamo uvodzeny, ale skusenost s velkymi datovymi rurovodmi a automatizaciou je silna.
**Pouzite odkazy:**
- Engineered and maintaining data extraction, processing, and visualization pipelines for network monitoring and performance analytics
- Developed ML-based predictive models (XGBoost) to identify anomalies in infrastructure telemetry, integrated with CI/CD pipelines
- Proven track record of building robust SQL- and Elastic-based data pipelines

### ✅ R10: Solid understanding of machine learning, deep learning, and MLOps methodologies
**Riziko/neistota:** Nesplnenie hlubokych ucenych (deep learning) ani MLOps metodologiou nie je priamo uvedene v textovych odkazoch, ale skusenost s ML je silna.
**Pouzite odkazy:**
- Machine Learning: AutoML, XGBoost, Scikit-learn, Predictive Analytics
- Developed ML-based predictive models (XGBoost) to identify anomalies in infrastructure telemetry
- AI-driven Infrastructure Anomaly Detection: Developed a Python + SQL + Grafana solution predicting call volume anomalies in real-time

### ✅ R11: Ability to explain technical concepts to non-technical audience
**Riziko/neistota:** Bez priameho uvodnenia publiku 'non-technical', ale kontext trenia inzhinierov a vedenia projektov naznacuje schopnost komunikacie.
**Pouzite odkazy:**
- Led internal technical sessions on automation, telemetry visualization, and Python data engineering for IT teams.
- Led global virtual teams (4–6 members) for network deployment and performance enhancement projects.
- Trained new engineers and served as SME in LAN/WAN technologies.

### ❌ R12: Enthusiasm for continuous learning and trying new approaches
**Riziko/neistota:** Poziadavka nie je podlozena ziadnym dokazom z CV.