H2HCLIP 
Title: Hybrid Semantic Optimization with Hierarchical Spatial-Wavelet Synergy for Zero-Shot Anomaly Detection
Abtract: Zero-shot anomaly detection with vision-language models offers a promising solution for privacy-constrained scenarios. However, the preference of pre-trained vision encoders for global semantic consistency, together with globally-aware prompts, introduces a bias toward object-centric holistic representations, while suppressing high- and low-frequency anomaly cues that preserve object-level semantics yet are critical for anomaly detection. In this paper, we propose H2HCLIP, which integrates hybrid semantic optimization with hierarchical spatial-wavelet synergy to explicitly enhance frequency-domain visual modeling while introducing fine-grained semantic guidance. For visual enhancement, we develop the frequency-aware refinement attention at intermediate encoder layers, amplifying deep-frequency patterns via Fourier transformation, while applying non-saliency anomaly completion and threshold-based filtering to accentuate anomalous representations. At the output layer, spatial-wavelet collaborative inference is conducted, where multiple wavelet transformations intricately merge local high-frequency subtleties with global low-frequency structures. For semantic completion, we generate image- and region-level prompts to enrich global-local semantics, complemented by sentiment-level texts that sharpen the decision boundaries through extreme semantic contrast. We leverage a feature-guided hybrid fusion mechanism and an intra-inter contrastive loss to facilitate the interaction of semantic information from region-to-whole, promoting the joint optimization of classification and segmentation. Extensive experiments on 14 real-world anomaly detection datasets reveal that H2HCLIP outperforms numerous methods.
<img width="1096" height="485" alt="image" src="https://github.com/user-attachments/assets/5c48ebd1-d3db-4dc1-9c06-1f0b49bf7e35" />
This repository provides the H2HCLIP code, including training and testing code, for easy reproduction.
For required packages, please refer to requirements.txt.
This method is based on AnomalyCLIP, and innovates upon it.
Dataset: For dataset and environment information, please refer to AnomalyCLIP, link: https://github.com/zqhang/AnomalyCLIP 
After downloading the mvtec AD dataset, please unzip the json_file folder and execute the mvtec.py file to obtain the dataset JSON file required by the model.
Generate the dataset JSON: Take MVTec AD for example (With multiple anomaly categories)
Structure of MVTec Folder:
mvtec/

│

├── meta.json

│

├── bottle/

│   ├── ground_truth/

│   │   ├── broken_large/

│   │   │   └── 000_mask.png

|   |   |   └── ...

│   │   └── ...

│   └── test/

│       ├── broken_large/

│       │   └── 000.png

|       |   └── ...

│       └── ...

│   

└── ...


We provide a script for the data in the json_file folder. Select the script and execute it. The generated JSON file stores all the information required for H2HCLIP.
python mvtec.py

Please extract the AnomalyCLIP_lib and mvtec_model compressed packages separately.

The test code for pixel-level results of the H2HCLIP method can be found in the test.py file. The execution command is as follows:
python test.py
