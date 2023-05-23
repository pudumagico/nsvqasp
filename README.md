# A Logic-based Approach to Contrastive Explaninability for Neurosymbolic Visual Question Answering

This repository contains the implementation of our ASP-based modular neurosymbolic framework for VQA 
and our abduction program for generating Contrastive Explanations.

## Requirements

The major software packages we used are:
1. [Python 3.8](https://www.python.org/)
2. [PyTorch 1.7.1](https://pytorch.org/)
3. [Clingo 5.6.2](https://potassco.org/clingo/)
3. [CUDA 12](https://developer.nvidia.com/cuda-downloads)

We suggest using [Conda](https://docs.conda.io/en/latest/) for packet management.
Simply install the listed packages and perform the steps described in [Setup](#setup). 
Our experiments where done on a system running Ubuntu 20.04.5 LTS.

## Project Structure

```
PROJECT_NAME
│   README.md
│   infer.py                    // Inference and optionally Abduction
│   gen_ce.py                   // Contrastive Explanation generator
│   dataset.py                  // Dataset and DataLoader
│   options.py                  // Options
│   tools.py                    // Auxiliary functions
│
└───language                    // Folder containing language models LSTM and Transformer (not implemented)
│   │   ...
│
└───vision                      // Folder containing YOLOv5 files
│   │   ...
│   
└───reasoning                   // Folder containing ASP Theory and Abduction programs.
    │   ...
```

## Setup

Set up a conda enviroment
`conda create -n NSVQASP`
Then install the requirements file
`conda install --file requirements.txt`

## Training

Please refer to the training instructions for [YOLOv5](https://github.com/ultralytics/yolov5) and the [LSTM](https://github.com/kexinyi/ns-vqa).


## Inference

The `main.py` program implements our model inference and optionally can produce contrastive explanations for wrong answers.
You can use this program via the following command:
`python infer.py --clevr_image_path <images_path> --clevr_scene_path <scene_path> --clevr_question_path <questions_path> --clevr_vocab_path <vocab_path> --vision_weights <YOLO_weights_path> --language_weights <LSTM_weights_path> --theory <theory_path>`
With the corresponding paths to each of the arguments.
To produce contrastive explanations when the model produces a wrong answer, add the following option
`--abduction <abduction_path>`

## Contrastive Explanations generator

To generate contrastive explanations for a particular instance, the user must provide a file 
containing the question, scene and foils to be considered as ASP facts described in the theory file.
Use the following command to generate explanations:
`python ce_gen.py --abduction_program <abduction_path> --theory_program <theory_path> --input_file <input_path>`
<!-- Additionally, the user may provide custom weights for the change rules in the abduction file by including 
weight facts of the form `weigth(change_operation, cost).`, where change_operation is one of the changes implemented in the abduction file and cost is the custom cost assigned to it.  -->


