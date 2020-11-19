## Physical Backdoors for Object Recognition

This simple script uses a custom dataset to train an object recognition model with a physical backdoor.

The 9 classes used in this object recognition dataset are: backpack, coffee mug, cell phone, laptop, purse, running shoe, sunglasses, tennis ball, and water bottle. 

The physical backdoor trigger is a smile emoji sticker.

### Requirements

`tensoflow >= 2.0`
`cuda >= 9.0`

### Using the code

1. Clone this repository wherever you choose.

2. Download the object recognition dataset __into the cloned folder__. The data can be found at this link: 

2. To train a physical backdoored model, simply run the command `python3 train.py`. You can add flags (specified in train.py) to customize the teacher model, etc. 