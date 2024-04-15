title: Flyte: AI In Production
use_katex: True
class: title-slide

![:scale 40%](images/flyte.svg)

# AI in Production

.larger[Thomas J. Fan]<br>
@thomasjpfan<br>

<a href="https://www.github.com/thomasjpfan" target="_blank"><span class="fa-brands fa-github"></span></a>
<a href="https://www.linkedin.com/in/thomasjpfan" target="_blank"><span class="fa-brands fa-linkedin"></span></a>
<a href="https://www.twitter.com/thomasjpfan" target="_blank"><span class="fa-brands fa-twitter"></span></a>
<a class="this-talk-link", href="https://github.com/thomasjpfan/odsc-east-2024-flyte" target="_blank">
Slides on Github: github.com/thomasjpfan/odsc-east-2024-flyte</a>

---

# Contents 📓

.g.g-middle[
.g-6[
- Flyte Overview
    - Production-Ready
    - Developer Experience
    - Scale
- Applications
]
.g-6.g-center[
![:scale 80%](images/flyte.svg)
]
]

---

.g.g-center.g-middle[
.g-6[
![:scale 80%](images/flyte.svg)
]
.g-6[
![:scale 80%](images/linux-foundation.svg)
]
]

---

# Two Personas

.g[
.g-6[
## Data Scientist 👩‍🔬
]
.g-6[
## Platform Engineer 👷‍♀️
]
]

---

class: chapter-slide

# Scale 🌎

???
- Built on Kubernetes
- Multi-tenancy
- Compute
    - Dynamic workflows
    - Map Tasks
    - Ray
    - Dask
    - Spark

---

class: chapter-slide

# Developer Experience 💻

???
- Strict typing
- Local and Remote execution
- Any language
- ImageSpec
- Declaractive Infrasturture
- Visualization

---

class: chapter-slide

# Production-Ready 🚀

???
- Versioned workflow
- Data Lineage
- Containers first
- Launchplans (Scheduling)
- Spot instances
- Intra-task Checkpointing

---

# Applications 💡

- Data Task
    - Loading data
        - BigQuery Agent
        - Snowflake Agent
        - SQL
        - DuckDB
    - Computation with data
        - Python
        - R
        - Julia
        - Scale:
            - Dask
            - Ray
            - Spark
        - Airflow Plugin
        - Databricks Agent
        - Data Validation/Quality
            - Great expectations
            - Pandera
    - Report/Dashboard/Images
    - Bioinformatices
        - ShellTask
        - Docker first
        - TypeTransformers
        - Union plugin
- Machine learning
    - Evaluate
        - Predictive models: Test data
        - Dashboard
    - Save as object
    - Human in the loop
    - Deploy
- AI Model
    - Finetuning, RAG
        - Pretrained model
    - Recommendation
        - Modalities
    - GPU
        - Selecting GPUs
        - Intra-task checkpointing
        - Spot instances
        - Scaling
            - PyTorch
            - Tensorflow
            - NVIDIA DGX Cloud

# Community 🌎

- Pluggable
    - Accerlated Datasets
    - NVIDIA DGX Cloud with Agents

- VSCode Plugin

---

# Closing

---

class: chapter-slide

# Appendix

???
- Metaflow
- Airflow
- Dagster
- Kubeflow
