# MSEB Interactive Walkthroughs (Google Colab)

Welcome to the hands-on workspace for
the **Massive Sound Embedding Benchmark (MSEB)**!

This directory contains interactive Google Colab notebooks designed to help
researchers, engineers, and developers understand the inner workings of
the MSEB pipeline.

Instead of configuring a large-scale local distributed environment, these
notebooks allow you to run and experiment with MSEB's architecture on a
single cloud VM directly through Colab.

## 🗺️ Recommended Path

If you are new to MSEB, we highly recommend going through the notebooks
in the following order to build your understanding step-by-step:

### 1. The Models: `encoder.ipynb`
Before evaluating anything, you need to understand how MSEB handles
AI models. This notebook walks you through the **Encoder Registry**.
You will see how MSEB standardizes, loads, and extracts embeddings
from various audio models (like Whisper) so they can all be evaluated fairly.

### 2. The Data: `svq.ipynb`
MSEB is incredibly rigorous about how it formats and grades data. Using
the **Simple Voice Questions (SVQ)** dataset as a case study, this notebook
demonstrates how MSEB handles data ingestion. Thanks to native streaming
support, you will see how MSEB processes audio metadata and ground-truth
transcripts on the fly without needing to download massive dataset archives
to your local disk.

### 3. The Pipeline: `benchmark.ipynb`
The grand finale. This notebook ties everything together into a complete
End-to-End Evaluation Pipeline (`Encoder → Runner → Task → Leaderboard`).
You will execute a full run, pass the predictions to the C++ math grader,
and see how to read the final JSON metric reports (like Word Error Rate).

---

## 💡 Important Note: Distributed vs. Single-Node Execution
MSEB is an enterprise-grade framework originally architected to process
massive audio dataset archives across distributed infrastructure using
Apache Beam.

To adapt this pipeline for a standard Colab instance, these notebooks
utilize a **single-node execution strategy**:

* **Dataset Streaming:** Instead of downloading hundreds of gigabytes of
raw data, we leverage MSEB's streaming capabilities to evaluate examples
on the fly.
* **DirectRunner:** We use Beam's `DirectRunner` to process the pipeline
on a single cloud machine rather than a distributed cluster.
* **Environment Adaptations:** We include a few minor Python runtime patches
(like enforcing specific C++ math extensions and standardizing module
namespaces) to ensure the pipeline runs smoothly in a standard Python 3.12
notebook environment.

These walkthroughs are designed to demonstrate the *mechanics* of the
pipeline. Once you understand the flow here, you will be fully prepared
to deploy MSEB natively across your own large-scale distributed clusters.

## 🚀 Getting Started
To run these notebooks, simply click the files above and look for
the **"Open in Colab"** button, or download the `.ipynb` files and upload
them to your own Jupyter/Colab environment.
