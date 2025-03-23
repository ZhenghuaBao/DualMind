# DualMind: A Novel Framework for Automated Image Contextualization and Meta-context Retrieval

[![License](https://img.shields.io/github/license/UKPLab/ukp-project-template)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

This repository is an extension of the **5Pillars framework** introduced by [Tonglet et al. (2024)](https://aclanthology.org/2024.emnlp-main.448). Our goal is to produce more **explainable** and **reliable** 5Pillars (5Pils) answers by incorporating essential contextual elements—such as **Source**, **Date**, **Location**, and **Motivation**—and enriching them with additional relevant information. Moreover, the framework introduces a **confidence score** to indicate the trustworthiness of each generated answer, supported by an explanation of any detected manipulations for improved transparency and interpretability.

Contact person: Zhenghua Bao ([zheng.hua.b@gmail.com](mailto:zheng.hua.b@gmail.com))

## Abstract

The rapid proliferation of visual misinformation highlights the critical need for effective automated tools to assist human fact-checkers. Recent advances, such as the “5 Pillars” framework, have introduced structured approaches for contextualizing images by identifying Provenance, Source, Date, Location, and Motivation. However, current automated methods face notable limitations: manipulation detection approaches largely neglect semantic discrepancies, evidence retrieval often misses critical supporting materials, and generated explanations frequently rely on irrelevant or biased information. To address these gaps, we propose a novel pipeline—**DualMind**, extending the original 5 Pillars framework with significant improvements: 

- A semantically-informed forgery detection method integrating SAM-based semantic segmentation, CLIP-ViT image features, and LLM-based reasoning; 

- Expanded evidence retrieval leveraging multiple reverse image search engines with evidence re-ranking using media bias and semantic relevance scores; 

- Contextual-aware answer generation; 

- A fallback pipeline employing keyword-based textual search when image-based retrieval fails. 

Experiments demonstrate that DualMind significantly improves evidence retrieval coverage, reduces misinformation in generated answers, and enhances the overall reliability of image contextualization. By systematically evaluating contextual reliability and forgery explanations, DualMind offers fact-checkers more robust and reliable references for detecting and debunking visual misinformation.

### Demo output

<p align="center">
  <img width="80%" src="figs/1.png" alt="header" />
</p>



## Create environment

```
$ conda create --name DualMind python=3.10
$ conda activate DualMind
$ pip install -r requirements.txt
$ python -m spacy download en_core_web_lg
```

### APIs

- Our baseline model relies on evidence retrieved with reverse image search using the [Google Vision API](https://cloud.google.com/vision/docs/detecting-web) and [Tineye](https://tineye.com/). All URLs for the text evidence that we used in our experiments are provided in this repo. However, should you want to collect your own evidence, you will need to create a Google Cloud account and create an API key, or use a different reverse image search service.

- To gather evidence using a keyword search, you'll need to generate a [Serper API Key](https://serpapi.com/). This key enables you to perform Google web searches and retrieve relevant textual and visual evidence based on the generated description of the provided image.

- If you want to use GPT4(-Vision) for answer generation, you will need an an API key from [OpenAI](https://platform.openai.com/api-keys).

## 5Pils dataset

We used 5pillars dataset provided by [Jonathan Tonglet](mailto:jonathan.tonglet@tu-darmstadt.de).

You can collect the images using the following script:

```
$ python scripts/build_dataset_from_url.py
```

### Collecting the evidence

1) Collect the text evidence based on their URLs:

```
$ python scripts/collect_RIS_evidence.py --collect_google 0 --evidence_urls dataset/retrieval_results/evidence_urls.json 
```

Instead of using the provided evidence set, you can also collect your own by setting *collect_google* to 1. This will require to provide a Google Vision API key. 

2) Collect the ebidence based on their keywords (Serper API required):

```
$ python scripts/collect_keyword_evidence.py --collect_keyword 1
```

Once the evidence has been collected, make sure to download the extracted images so they can be used later for similarity checks:

```
$ python scripts/download_keyowrd_search_images.py
```

If you want to validate downloaded images:

```
$ python scripts/validate_images.py
```

This will display a list of images that failed to download correctly.

###  Compute embeddings 

Compute embeddings for evidence ranking based on images or context and similarity checks:

```
$ python scripts/get_embeddings.py
```

## Pipeline

<p align="center">
  <img width="80%" src="figs/2.png" alt="header" />
</p>

You can generate answers for a specific pillar. The plus pillar includes a **confidence score** and **forgery explanation**. For example, to generate answers for the **Date** pillar using multimodal LLaVA:

a) Original baseline:

```
$ python scripts/get_5pillars_answers.py --results_file output/results_date.json --task date --modality multimodal --model llava
```

a) Core Pipeline:

```
$ python scripts/generate_core_pipeline_answers.py --results_file output/core_results_date.json --task date --modality multimodal --model llava
```

a) Fallback Pipeline:

```
$ python scripts/generate_fallback_answers.py --results_file output/fallback_results_date.json --task date --modality multimodal --model llava
```

### Answer Selection

After generating both core and fallback answers, you can choose the final answers by specifying the task, e.g. for the **Date** pillar, using the following command:

```
$ python scripts/answer_selection.py --task date
```

This will produce a JSON file containing the final output of the pipeline.

## Evaluation

Assess the model's performance using the final results. For instance, when evaluating the **Date** pillar:

```
$ python scripts/evaluate_answer_generation.py --results_file output/final_results_date.json --task date
```
 

## Citation

If you use this work or the underlying dataset, please cite the original 5Pillars paper:

```bibtex 
@inproceedings{tonglet-etal-2024-image,
    title = "{``}Image, Tell me your story!{''} Predicting the original meta-context of visual misinformation",
    author = "Tonglet, Jonathan  and
      Moens, Marie-Francine  and
      Gurevych, Iryna",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.448",
    pages = "7845--7864",
}
```



## Disclaimer

> This project was developed as part of the **Project Lab: Multimodal Artificial Intelligence 2024** at **TU Darmstadt**. It builds heavily upon the foundational work ["Image, Tell me your story!" Predicting the original meta-context of visual misinformation](https://aclanthology.org/2024.emnlp-main.448/).
>
> We sincerely thank the original authors for sharing their dataset and answering our questions. Special thanks to our mentor **Mark** for his insightful guidance and support throughout the course.
