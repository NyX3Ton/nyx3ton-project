# NyX3Ton Lab / Project Hub

Welcome to my main GitLab project space.  
This repository acts as a central hub for my work in IT infrastructure, data analytics, AI/ML, automation, monitoring, SQL databases, telecom analytics, and homelab engineering.

The goal of this space is not only to store code, but to build practical solutions that connect real systems, data pipelines, observability, automation, and intelligent analysis.

---

## Main Focus Areas

| Area | Description |
|---|---|
| AI / Machine Learning | XGBoost, Optuna, NLP models, embeddings, sentence transformers, candidate validation |
| Data Engineering | ETL pipelines, ElasticSearch, MS SQL, PostgreSQL, pandas, SQLAlchemy |
| Monitoring & Observability | Grafana, Kibana, Elastic Stack, Logstash, Metricbeat, SNMP telemetry |
| Telecom / SBC Analytics | Oracle SBC, Ribbon SBC, SNMPv3, CDR analysis, SIP error monitoring |
| Business Intelligence | Power BI, Grafana dashboards, KPI reporting, data models |
| Infrastructure / Homelab | Proxmox, Ubuntu Server, TrueNAS, Tailscale, RDP, Docker, Linux services |
| Automation | Python scripts, self-healing processes, scheduled jobs, alerting logic |

---

## Featured Projects

### 1. CV Validator / AI Candidate Validator

An AI-powered application for analyzing CVs, comparing candidates against job descriptions, and evaluating candidate suitability.

**Technologies:**

- Python
- Sentence Transformers
- HuggingFace models
- Zero-shot classification
- Semantic matching
- Gradio / Streamlit
- pandas

**Key Features:**

- CV information extraction
- Candidate-to-job matching
- Semantic similarity scoring
- Candidate suitability assessment
- Explainable and auditable HR output

Repository: `[CV Validator](./CV-Validator)`

---

### 2. HR Attrition AI / Employee Risk Analytics

A machine learning solution for predicting employee attrition risk and analyzing the factors that influence employee turnover.

**Technologies:**

- Python
- XGBoost
- Optuna
- scikit-learn
- SQL Server
- PostgreSQL
- Power BI
- Gradio
- SHAP / feature importance

**Key Features:**

- Employee attrition prediction
- Hyperparameter tuning with Optuna
- SQL database integration
- HR analytics dashboards
- Employee risk segmentation
- Feature importance and explainability

Repository: `[HR Attrition AI](./HR-Attrition-AI)`

---

### 3. SBC SNMP Collector

A Python-based SNMPv3 collector for gathering telemetry from Oracle and Ribbon SBC devices and storing the results in Microsoft SQL Server.

**Technologies:**

- Python
- pysnmp
- asyncio
- Microsoft SQL Server
- SQLAlchemy
- ODBC Driver 18 for MS SQL
- Rotating logs

**Key Features:**

- Vendor detection based on `sysObjectID`
- Oracle / Sonus / Acme Packet and Ribbon SBC support
- SNMP profile fallback logic
- Vendor-specific SQL table writes
- Self-healing restart logic
- Rotating logs and detailed diagnostics

Repository: `[SBC SNMP Collector](./SBC-SNMP-Collector)`

---

### 4. Elastic / Logstash / Grafana Monitoring

A monitoring and observability lab built around Elastic Stack, Logstash, Metricbeat, SNMP, syslog, and Grafana.

**Technologies:**

- ElasticSearch
- Logstash
- Kibana
- Metricbeat
- Grafana
- SNMP
- Syslog
- Linux systemd

**Key Features:**

- Centralized syslog ingestion
- SNMP polling
- System metrics collection
- Grafana dashboards
- CPU, memory, availability, and service monitoring
- Watchdog-style monitoring logic

Repository: `[Elastic Monitoring Lab](./Elastic-Monitoring-Lab)`

---

### 5. Telecom CDR Anomaly Detection

An anomaly detection pipeline for analyzing SIP 500 errors from telecom CDR data.

**Technologies:**

- Python
- ElasticSearch
- pandas
- XGBoost
- Optuna
- Microsoft SQL Server
- Grafana

**Key Features:**

- CDR extraction from ElasticSearch
- SIP 500 error filtering
- Daily aggregation by site, device, and user
- Expected error volume prediction
- Anomaly detection for unusual error spikes
- SQL output for aggregated and detailed results
- Grafana visualization layer

Repository: `[CDR Anomaly Detection](./CDR-Anomaly-Detection)`

---

### 6. Homelab Infrastructure

Documentation and configuration notes for my own lab environment built around Proxmox, Ubuntu Server, VPN access, monitoring, and self-hosted services.

**Technologies:**

- Proxmox VE
- Ubuntu Server
- Tailscale
- TrueNAS / Samba
- RDP
- Elastic Stack
- Docker
- systemd

**Key Features:**

- Virtualized lab services
- Secure remote access over VPN
- Server monitoring
- Centralized logging
- Self-hosted service testing
- Separation between lab, test, and production-like workloads

Repository: `[Homelab Infrastructure](./Homelab-Infrastructure)`

---

## Tech Stack

### Languages

- Python
- SQL
- Bash
- PowerShell
- Markdown

### Databases

- Microsoft SQL Server
- PostgreSQL
- ElasticSearch

### AI / Machine Learning

- XGBoost
- Optuna
- scikit-learn
- pandas
- numpy
- SHAP
- HuggingFace Transformers
- Sentence Transformers

### Monitoring & Observability

- Grafana
- Kibana
- Logstash
- Metricbeat
- SNMP
- Syslog

### Infrastructure

- Proxmox
- Ubuntu Server
- Windows Server
- Tailscale
- Docker
- GitLab CI/CD

---

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.com/nyx3ton-group/nyx3ton-project.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.com/nyx3ton-group/nyx3ton-project/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/user/project/merge_requests/auto_merge/)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***
