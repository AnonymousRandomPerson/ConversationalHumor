# ConversationalHumor
## Overview
This is a chatbot that employs humor within the context of two-person conversations (as opposed to more isolated humor like jokes and puns). It is built off of the seq2seq code found in the [DeepQA](https://github.com/Conchylicultor/DeepQA) repository by [Conchylicultor](https://github.com/Conchylicultor). The basis of the chatbot is a pretrained seq2seq model on the Cornell movie dialogues corpus, provided by Nicholas C. and located [here](https://drive.google.com/drive/folders/0Bw-phsNSkq23c29ZQ2N6X3lyc1U).

## Requirements
* [Python](https://www.python.org/) 3
* [TensorFlow](https://www.tensorflow.org/) 0.12.1
* [NLTK](https://www.nltk.org/)
* [SciPy](https://www.scipy.org/) (including NumPy)
* [Send2Trash](https://github.com/hsoft/send2trash)
* [tqdm](https://github.com/tqdm/tqdm)

Note that this repository uses Git submodules to link to other repositories. After cloning this repository, the submodules can be retrieved with the following commands.

    git submodule init
    git submodule update

## Usage
First, change into the humor folder.

    cd humor

To train the model to select between using or not using humor (chatting with another chatbot that always picks a "normal" response), run the following command.

    python -m qlearn --modelTag cornell --rootDir DeepQA --test daemon -f evaluated_rl

* *-f evaluated_rl* (*--model-file*) uses a pre-existing model for the selector. *evaluated_rl* can be changed to a different name to train a new model.
* *-d* (*--debug-print*) will print more details about the chatbot's decisions, such as q-values for selecting which internal chatbot to use, the internal chatbot that was selected, and the "normal" response if the chatbot chose to use the humorous chatbot.
* *-t* (*--rl-test*) will test the model instead of training it.
* *-i* (*--interactive*) will allow you to directly converse with the chatbot instead of watching it converse with another chatbot.
* *-o* (*--other-show*) will show the response of the chatbot option that was not chosen to be used.
* *-s* (*--reduce-swear*) will reduce the usage of certain swear words by the humor chatbot.