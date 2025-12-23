# textt

Project to the subject POVa. Automatic text transcription.

## Dataset

We used [Handwriting Adaptation Dataset](https://pero.fit.vutbr.cz/handwriting_adaptation_dataset)
from [Pero Project](https://pero.fit.vutbr.cz/). To be able to train the model,
you should download it from the above link.

## Usage

To start training the model, you can run (note that the dataset is expected
to be in the same format as the dataset from Pero mentioned above):

```bash
./main.py train -d path/to/dataset
```

You can also set number of epochs, batch size or continue training already
existing model. The training is automatically stored to the output folder,
which you can also set:

```bash
./main.py train -d path/to/dataset -e 67 -b 32 -m model.pt -o output/path
```

If you want to transcribe a text, all you need is the trained model and image
you want to transcribe. This will print out the predicted text:

```bash
./main.py run -i path/to/image.jpg -m path/to/model.pt
```
