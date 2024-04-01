# Flyte: A Production-Ready Open Source AI Platform

*By Thomas J. Fan*

[Link to slides](https://thomasjpfan.github.io/odsc-2024-east-flyte/)

Quick and sustainable iteration cycles from development to production are crucial to maximizing the impact of our AI and ML models. Flyte is a Linux Foundation open-source production-ready platform for AI, machine learning, and analytics workflows. For the ML practitioner and AI engineer, Flyte allows us to debug locally, execute remotely at scale, and deploy the same code into production. Flyte's Python library, flytekit, enables us to utilize Python's vast ecosystem of data science libraries in our workflows. For platform engineers, Flyte is built on Kubernetes and has multi-tenancy at its core, which is vital for sharing and managing resources effectively.

In this talk, we learn about Flyte's key features and benefits through demonstrations of specific AI and machine learning workflows. Specifically, we explore workflows for fine-tuning large language models, machine learning with tabular and image datasets, and bioinformatics. These examples demonstrate how Flyte's features, such as data lineage, caching, and strong typing inference, are used in different contexts. Lastly, we explore Flyte's plugins, enabling us to interact with other libraries in the data science ecosystem. For example, the PyTorch and Tensorflow plugins allow us to run distributed AI training jobs. The Dask, Ray, and Spark plugins help us scale out computation with a compute cluster. By the end of this talk, you'll have a good overview of what Flyte is capable of and how it can fit into your AI and ML workflows.
