All the code and report files can be downloaded from our repository: https://github.com/fnoya/CS445-Colorization-Project/

The code is organized in the following files:

- Project.ipynb 				Jupyter notebook with the colorization models, design and validation.
- Project.py					Python script to train the colorization models on GPU machines.
- deeplabv3_segmentation.ipynb	Jupyter notebook with the segmentation model and results.

The models weights that are required for some of the scripts are in the "models" folder that can be downloaded from https://drive.google.com/file/d/1yOtb3xmGeRKZwLcJsiE8arKqJGXFtwqO/view?usp=sharing.
	- net_G_resnet18_model-19.pt	Results of pretraining the Resnet18 generator.
	- colorization2-epoch20.pt		GAN with UNet-Resnet18 generator and patch discriminator, trained for 20 epochs.
	- colorization6-ViT-epoch20.pt	ViT-LSGAN final model, trained for 20 epochs.


The data we used can be found in the following folders:

- Historic images: https://1drv.ms/u/s!Ao7kvuRfJCHQlpVPcmu6XpQhbzDehA?e=ynrEUK
- Old portraits: https://1drv.ms/u/s!Ao7kvuRfJCHQlpdcSEt_ylPe1n2DaQ?e=fYmNlv
- Own photographs: https://1drv.ms/u/s!Ao7kvuRfJCHQlPZ9bnEX2Js6EutM3g?e=QNWXji
- Results from colorization: https://1drv.ms/u/s!Ao7kvuRfJCHQlpV4ChIptU613kUcmw?e=iDQGmO
- Results from segmentation: https://1drv.ms/u/s!Ao7kvuRfJCHQlpp6mmCSaLBXkspc5w?e=jOW1ZR