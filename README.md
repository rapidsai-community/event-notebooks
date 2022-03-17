# RAPIDS Event Notebooks
---
## Table of Contents
* [Intro](#intro)
* [Exploring the Repo](#exploring)
* [Additional Resources](#more)
  
---

## Introduction <a name="intro"></a>

Welcome to the RAPIDS Event Notebooks Repo!

### RAPIDS Event Notebooks
Here is the collection of RAPIDS notebooks and script that the RAPIDS team have presented at Conference and Meet up events.

These notebooks are built by the RAPIDS team but will not be maintained.  These notebooks are guaranteed to only run using the RAPIDS version that was current and stable when the notebook was released, unless otherwise noted. Please be prepared to update any notebook if you plan to make it run on today's latest.
 

### How to Contribute <a name="contributing"></a>

Only RAPIDS team members can contribute to RAPIDS Event Notebooks.  

If you'd like to contribute a notebook that you presented at an event, please check out our [Community Notebooks](https://github.com/rapidsai-community/notebooks-contrib) and either:
1. make a PR into the `community_events` folder if you plan to maintain the notebook, 
1. into our `the_archive\archived_rapids_event_notebooks` folder if you don't.  Unmaintained notebooks will be moved into this folder 

If you're a RAPIDS Team member, please create a PR and notify @taureandyerNV when you're ready for a review.

## Exploring the Repo <a name="exploring"></a>

Each folder named for a conference or event.  Each folder contains the notebooks and scripts from the Conference or Meet up that it was presented at.  Below is a index of all the notebooks.

### Event Notebooks List
Click each event to expand and view the notebooks within

<details>
  <summary>GTC Spring 2022</summary>
  
  * [Link to Folder](event_notebooks/GTC_Spring_2022/numerical_computing) 
    * [Single Threaded & Single GPU Methods](event_notebooks/GTC_Spring_2022/numerical_computing/single-cpu-gpu.ipynb) 
    * [Parallel CPU with Numba](event_notebooks/GTC_Spring_2022/numerical_computing/multi-cpu-numba.ipynb)
    * [Multi-GPU with Dask cuDF + Numba CUDA](event_notebooks/GTC_Spring_2022/numerical_computing/multi-gpu-dask-cudf-numba.ipynb)
    * [Multi-GPU with Threading + RMM + Numba CUDA](event_notebooks/GTC_Spring_2022/numerical_computing/multi-gpu-threading-rmm-numba.ipynb)

</details>

<details>
  <summary>GTC Spring 2021</summary>
  
  * [Link to Folder](event_notebooks/GTC_2021/credit_scorecard) 
    * [WOESC Demo Vehicle Data](event_notebooks/GTC_2021/credit_scorecard/cpu/woesc_demo_vehicle_data.ipynb) 
    * [XGBSC Demo Vehicle Data](event_notebooks/GTC_2021/credit_scorecard/cpu/xgbsc_demo_vehicle_data.ipynb)

</details>

<details>
  <summary>JupyterCon 2020</summary>  
 
  * [Link to Folder](event_notebooks/JupyterCon_2020_RAPIDSViz)
    * [00 Index and Introduction](event_notebooks/JupyterCon_2020_RAPIDSViz/00%20Index%20and%20Introduction.ipynb) 
    * [01 Data Inspection and Validation](event_notebooks/JupyterCon_2020_RAPIDSViz/01%20Data%20Inspection%20and%20Validation.ipynb)
    * [02 Exploratory Data Visualization](event_notebooks/JupyterCon_2020_RAPIDSViz/02%20Exploratory%20Data%20Visualization.ipynb)
    * [03 Data Analysis with Visual Analytics](event_notebooks/JupyterCon_2020_RAPIDSViz/03%20Data%20Analysis%20with%20Visual%20Analytics.ipynb) 
    * [04 Explanatory Data Visualization.ipynb](event_notebooks/JupyterCon_2020_RAPIDSViz/04%20Explanatory%20Data%20Visualization.ipynb)

</details>

<details>
  <summary>KDD 2020</summary>
  
  * [Link to Folder](event_notebooks/KDD_2020)
    * [Seattle Parking Notebooks](event_notebooks/KDD_2020/notebooks/parking/)
      * [1) RAPIDS Seattle Parking](event_notebooks/KDD_2020/notebooks/parking/codes/1_rapids_seattleParking.ipynb) 
      * [2) RAPIDS Seattle Parking Graph](event_notebooks/KDD_2020/notebooks/parking/codes/2_rapids_seattleParking_graph.ipynb)
      * [3) RAPIDS Seattle Parking Nodes](event_notebooks/KDD_2020/notebooks/parking/codes/3_rapids_seattleParking_parkingNodes.ipynb)
    * [Rossmann Store Sales Example](event_notebooks/KDD_2020/notebooks/nvtabular/rossmann-store-sales-example.ipynb) 
    * [cyBERT Training Inference](event_notebooks/KDD_2020/notebooks/cybert/cyBERT_training_inference.ipynb)
    * [NYCTaxi Notebooks](event_notebooks/KDD_2020/notebooks/Taxi)
      * [NYCTaxi](event_notebooks/KDD_2020/notebooks/Taxi/NYCTax.ipynb)
    * [Single-Cell RNA-seq Analytics](event_notebooks/KDD_2020/notebooks/Lungs)
      * [RAPIDS & Scanpy Single-Cell RNA-seq Workflow](event_notebooks/KDD_2020/notebooks/Lungs/hlca_lung_gpu_analysis.ipynb)

</details>

<details>
  <summary>TMLS 2020</summary>
  
  * [Link to Folder](event_notebooks/TMLS_2020/notebooks/Taxi)
    * [Overview-Taxi](event_notebooks/TMLS_2020/notebooks/Taxi/Overview-Taxi.ipynb)

</details>

---

## Additional Resources <a name="more"></a>
- [Visit our RAPIDS Showcase](https://github.com/rapidsai-community/showcase) for our demonstration notebooks built by the RAPIDS team and our partners.
- [Visit our Community Notebooks repo](https://github.com/rapidsai-community/notebooks-contrib) for notebooks built by the RAPIDS team, our Ecosystem Partners, and RAPIDS users like you!
- [Watch our Youtube Channel](https://www.youtube.com/channel/UCsoi4wfweA3I5FsPgyQnnqw/featured?view_as=subscriber) or see [list of videos](multimedia_links.md) by RAPIDS or our community.  Feel free to contribute your videos and RAPIDS themed playlists as well!
- [Read our Blogs on Medium](https://medium.com/rapids-ai/)
- [Listen our Podcast, RAPIDSFire](https://anchor.fm/rapidsfire)
