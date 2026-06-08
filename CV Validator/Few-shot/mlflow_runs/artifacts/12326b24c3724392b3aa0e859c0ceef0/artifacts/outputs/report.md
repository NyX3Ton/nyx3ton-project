# AI CV validator

**Pozicia:** AI Engineer
**Seniorita:** unknown
**Zdroj poziadaviek:** llm_json_plus_fallback_merge
**Celkove skore:** 47.99 / 100
**Odporucanie:** Ciastocne vhodny / vyzaduje manualne posudenie

> Vystup je odporucanie pre cloveka, nie automaticke rozhodnutie o kandidatovi.

## Anonymizovany profil kandidata
Nepodarilo sa spolahlivo extrahovat profil.

## Vyhodnotenie poziadaviek

| Stav | Poziadavka | Skore | Priorita | Vysvetlenie |
|---|---|---:|---|---|
| ✅ splnene | Python | 85 | must / w=5.0 | CV obsahuje priame spominyanie pouzitia Pythona v kontexte backendovych pipelineov a ML modelov. |
| ✅ splnene | SQL | 85 | must / w=5.0 | CV obsahuje priame spominy MS SQL a SQL v kontexte vyvoja riešení. |
| ❌ nesplnene | REST APIs | 0 | must / w=4.0 | V dodanych odkazoch z CV nie je zisteny priamy dokaz o znalosti REST APIs. |
| ✅ splnene | LLM technológie a AI capabilities | 85 | must / w=4.0 | Kandidat má priamu skusenost s AI/ML modelami, prediktivnou analytikou a patentovanou aplikaciou AI v infrastrukture. |
| ✅ splnene | Git | 85 | must / w=3.0 | Kandidat spomina integraci s CI/CD pipeline v GitLab, co naznacuje zavislost na Git. |
| ❌ nesplnene | Google Cloud Platform | 0 | must / w=3.0 | V dodanych odkazoch z CV nie je ziadny dokaz o znalosti Google Cloud Platform. |
| ❌ nesplnene | vysokoškolské vzdelanie | 0 | must / w=3.0 | Vzdelanie je stredne (Stredna zivnostenska skola), nie vysokoškolske. |
| ✅ splnene | analytické myslenie | 85 | must / w=3.0 | Kandidat explicitne uvádza skusenosti s analytickym myslenim, datovym navrhom a analytikou v pracovnom popise. |
| ❌ nesplnene | skúsenosti s GCP services | 0 | nice / w=2.0 | V dodanych odkazoch z CV nie je ziadny dokaz o skusenostiach s GCP services. |
| ❌ nesplnene | skúsenosti s e-commerce platformami | 0 | nice / w=2.0 | V dodanych odkazoch z CV nie je ziadny dokaz o skusenostiach s e-commerce platformami. Kandidat pracuje v oblasti telekomunikacii a siete. |
| ❌ nesplnene | zvedavosť a rýchle učenie sa | 0 | nice / w=2.0 | V dodanych odkazoch z CV nie je zistena zvedavost ani skusenost s ucenim sa. |
| ✅ splnene | vzdelanie v oblasti IT, AI alebo analýzy dát | 85 | nice / w=2.0 | Kandidat má silne prepojené skusenosti s AI, ML a analýzou dát (Python, SQL, XGBoost, patenty) aj vzdelanie v oblasti elektrotechniky a manažmentu, čo spĺňa kritérium IT/AI/analýzy dát. |
| 🟡 ciastocne_splnene | Tvorba Python automatizácií v Google Cloud Platform a SQL transformácií v BigQuery pre spracovanie zákazníckych požiadaviek | 65 | unknown / w=1.5 | Kandidat má silne skusenosti s Python a SQL a vytvaram automatizovanych pipelineov a spracovania datovych tokov. Pouziva MS SQL a graficke dashboardy (Grafana, Power BI). Chybajuci je priamy dokaz o Google Cloud Platform alebo BigQuery, ktore su v zadanom texte absentne. |
| ❌ nesplnene | Implementácia API integrácií s e-commerce platformou, CRM, logistickými a inými systémami; | 0 | unknown / w=1.5 | CV obsahuje skusenosti s databázovými pipeline, monitorovaním a automatizaciou, ale nie je ziadny priamy dokaz o implementácii API integracií s e-commerce, CRM alebo logistikou. |

## Odkazy a poznamky

### ✅ R1: Python
**Riziko/neistota:** Uroven skusenosti nie je precisne definovana, ale pritis je jasny.
**Pouzite odkazy:**
- Engineered and maintaining data extraction, processing, and visualization pipelines for network monitoring and performance analytics (MS SQL, Python, Logstash, Grafana).
- Developed ML-based predictive models (XGBoost) to identify anomalies in infrastructure telemetry, integrated with CI/CD pipelines in GitLab.

### ✅ R2: SQL
**Pouzite odkazy:**
- MS SQL
- Python + SQL + Grafana solution

### ❌ R3: REST APIs
**Riziko/neistota:** Kandidat spomina vyvoj REST API v predchadzajucich inzeratoch, ale v tomto texte sa vyskytuje len Python, SQL, Grafana a iné techniky bez explicitnej zmieny o REST API.

### ✅ R4: LLM technológie a AI capabilities
**Riziko/neistota:** Hoci je kandidát skusenost s AI, explicitne spomienanie LLM (Large Language Models) v textoch nie je priradene, len obecné AI/ML a ML modely.
**Pouzite odkazy:**
- Developed ML-based predictive models (XGBoost)
- AI-driven Infrastructure Anomaly Detection: Developed a Python + SQL + Grafana solution predicting call volume anomalies in real-time
- Patent Activity: Filed multiple patents in AI/ML applied to IT infrastructure and automation analytics

### ✅ R5: Git
**Riziko/neistota:** Bez priameho uzitia slova Git, ale silny indikativny dokaz.
**Pouzite odkazy:**
- integrated with CI/CD pipelines in GitLab

### ❌ R6: Google Cloud Platform
**Riziko/neistota:** Kandidat uvádza skusenosti s Cisco, Oracle a vlastnym Pythonom, ale bez explicitneho spomenu Google Cloud Platform.

### ❌ R7: vysokoškolské vzdelanie
**Riziko/neistota:** Kandidat nesplni podmienku vysokoškolskeho vzdelania.

### ✅ R8: analytické myslenie
**Pouzite odkazy:**
- data-driven infrastructure design
- performance analytics
- predictive models

### ❌ R9: skúsenosti s GCP services
**Riziko/neistota:** Kandidat spomina len Cisco, Oracle a MS SQL, GCP nie je uvodeny.

### ❌ R10: skúsenosti s e-commerce platformami
**Riziko/neistota:** Poziadavka sa nepodarilo podlozit ziadnym dokazom z CV.

### ❌ R11: zvedavosť a rýchle učenie sa
**Riziko/neistota:** Poziadavku sa nepodarilo podlozit ziadnym dokazom z CV.

### ✅ R12: vzdelanie v oblasti IT, AI alebo analýzy dát
**Riziko/neistota:** Vzdelanie je uvedené ako študentstvo na fakulte manažmentu, ale skusenosti a projekty jasne potvrdzujú prax v oblasti IT a AI.
**Pouzite odkazy:**
- AI-driven Infrastructure Anomaly Detection: Developed a Python + SQL + Grafana solution predicting call volume anomalies in real-time
- Patent Activity: Filed multiple patents in AI/ML applied to IT infrastructure and automation analytics
- Core Skills • Programming & Data: Python, Pandas, Numpy, SQL, Polars, Power BI, Grafana, Kibana

### 🟡 R13: Tvorba Python automatizácií v Google Cloud Platform a SQL transformácií v BigQuery pre spracovanie zákazníckych požiadaviek
**Riziko/neistota:** Chyba priameho spomenutia Google Cloud Platform a BigQuery v CV. Skusenosti sa vzaju s podobnymi toolmi (MS SQL, Grafana) a automatickou spracovaniom, ale platforma nie je potvrdena.
**Pouzite odkazy:**
- Engineered and maintaining data extraction, processing, and visualization pipelines for network monitoring and performance analytics (MS SQL, Python, Logstash, Grafana)
- Developed a Python + SQL + Grafana solution predicting call volume anomalies in real-time
- Core Skills: Python, SQL, Power BI, Grafana, Kibana

### ❌ R14: Implementácia API integrácií s e-commerce platformou, CRM, logistickými a inými systémami;
**Riziko/neistota:** Chyba priameho dokazu o vyžadovanych integraciach.