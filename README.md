# ASVspoof-Polynomial-Model

ASVspoof Polynomial Networks (NCD)
This repository contains code and Jupyter notebooks for training and evaluating Polynomial Networks (PNs) using the Nested Couple Decomposition (NCD) technique for the ASVspoof challenge.

### Environmnet
* Python == 3.8.19
* PyTorch == 2.4.1
* Required Python packages (listed in requirements)
* ASVspoof dataset

In the premise of this project we leverage 2 repositories, the ASVspoof and RawBoost. To install most dependencies automatically (you can also follow the instructions from these repo): 

    Python
    pip install -r requirements_asv.txt
    pip install -r requirements_raw.txt


## Step 1: Clone the ASVspoof repository
    
    Bash
    git clone https://github.com/asvspoof-challenge/2021


## Step 2: Apply changes to the ASVspoof repository
Some things to pay attention from this repository are (Don't worry, this repository also includes a version of the ASV_spoof with all the necessry changes so you can train your own models):
1) Only the LA/Baeline-LFCC-LCNN subdirectory contains the actual "core_scripts" and "core_modules" folders. These folders contain multiple python scripts that manage various tasks in the background.
2) The DATA folder contains a single ".gitkeep" in order to be succesfully created (even though empty) during cloning of the repo. Howeveve, in this file you should put your own dataset with the correct format ( see the "toy_dataset.zip" in https://www.dropbox.com/sh/gf3zp00qvdp3row/AABc-QK2BvzEPj-s8nBwCkMna/temp/asvspoof2021/toy_example.tar.gz )
3) The most importnat scrpts you will need to manage during training are the **config.py**, **main.py** and **model.py** ( all are located under "baseline_DF/project" )
4) The first time you train a model, the repo will create 3 dictionaries containing metadata of your files. If in the future you change your data (here for augmentation reasons) you will also need to ***DELETE*** these dictionaries befoe start trainig again.
5) The execution of the repo start by running the bash script "02_toy_example.sh" in a Terminal. If you DO NOT INTENT to use a pre-trained model (as done in this repo) you must change the order in which the training and evalulation are done here (first execute the **"00_train.sh"** and then the **"01_eval.sh"**). In adittion to that, the repo will save by default the checkpoint of your trained model, under the "baseline_LA" folder. The "eval.sh" uses that checkpoin in order to start the inference stage. Make sure to correct the path (From trained_model=__pretrained/trained_network.pt to wherever you choose to save your model)
6) In the "config.py" make sure to set the correct paths to the flac files (for train/dev/eval) under the DATA folder
7) Here the development dataset is not used for hyper-parameter tuning but for evaluating the model after each epoch and keep a record of best-epochs for the early stopping mechanism


## Step 3: Data Preparation
Download the ASVspoof dataset and follow the instuctions in ASVspoof_Guide.ipynb in order to make the neccesary changes and create the files needed for training


## Step 4: Training
You can pick one of the listed models below to run
1) Default AVSspoof Conv2D (with logits extraction): <code style="color : name_color">model_conv2D_logits.py</code>
2) Multi-degree Polynomial: <code style="color : name_color">model_MultiDegreePolynomial.py</code>
3) Single Polynomial (NCP): <code style="color : name_color">model_ncp.py</code>
4) NCP + Special layer: <code style="color : name_color">model_ncp_special.py</code>
5) NCP + Special layer + More Convolution Layers + More BLSTM Layers: <code style="color : name_color">model_ncp_special_extra.py</code>


## Step 4: Evaluation
The model uses the development dataset to determine the best epoch and extracts the log_eval_score.txt from the evaluation dataset at that epoch. These scores are then evaluated against the ground truth metadata to calculate false positives and false negatives. Two models are implemented: one from the AVspoof repo with a cost model (false positives are more costly than false negatives) and another without a cost model. Both models calculate the EER (Equal Error Rate), with the first using t-CDF and the second using a different method.



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
Below are some URL you might find helful during your journey!

* [Polynomial Networks](https://github.com/polynomial-nets/tutorial-2022-intro-polynomial-nets)
* [Train/Dev Dataset 2019 (LA.zip)](https://datashare.ed.ac.uk/handle/10283/3336)
* [Eval Dataset 2021 (Speech_Data/LA subset)](https://zenodo.org/records/4837263)
* [Eval Dataset 2021 (LA keys and metadata)](https://www.asvspoof.org/index2021.html)
* [RawBoost Repository](https://github.com/TakHemlata/SSL_Anti-spoofing)
* [Repo-Viualizer](https://github.com/githubocto/repo-visualizer)
* [README Guide](https://github.com/othneildrew/Best-README-Template)



## Citation
If you find this work helpful, please cite it in your publications.

    @misc{Psaltis2024,
      author = {Psaltis, P. Stelios},
      title = {Project Title},
      year = {2024},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/PsaltisStelios/ASVspoof_Polynomial_Model}}
    }



  <p align="right">(<a href="#readme-top">back to top</a>)</p>
