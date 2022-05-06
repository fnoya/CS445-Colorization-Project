\title{Selective Coloring}
\author{Francisco Noya\\ \texttt{fnoya2@illinois.edu}\\
\And Mesay Taye\\ \texttt{mesayst2@illinois.edu} \\
}
\date{2022-05-06}

In this project we implemented a tool to selectively and automatically colorize the foreground of black and white photographs.  The tool consists of two main algorithms.  First,  a ViT-LSGAN network colorizes the entire photograph.  Second, a segmentation network segments the foreground from the background.  Lastly, colorized foreground is blended with the original background to produce a distinctive image.

# Colorization Network

## Approach
For the colorization task we worked on the __Lab__ colorspace.  In this space, the __L__ channel contains the luminance or intensity information, while the __ab__ channels contain the color information.  Therefore, a neural network can be trained with the __L__ channel of regular color images as input. Its predictions will be "fake" __ab__ channels and its loss will be calculated with the "real" __ab__ channels.

After a short bibliographic review we found that although traditional convolutional neural networks (CNNs) could produce results almost indistinguishable from real color photos [@ZhangIE16], _Generative Adversarial Networks_ or GANs [@goodfellow2014] were the most proper approach for this kind of problem.  This network architecture contains two modules, a Generator and a Discriminator.  Both models are trained in paralell.  The objective of the generator is to produce outputs similar enough to the ground truth that can fool the discriminator.  The discriminator's objective is to properly tell the ground truth from the discriminator output.

Since training this kind of networks requires large datasets and computing time, we decided to use pretrained models that have been used for other tasks such as object classification.  We followed a tutorial inspired by the _pix2pix_ paper [@pix2pix] but instead of training a naïve _UNet_ as the generator, it used a _ResNet18_ network as the generator [@colorizationtutorial].  Similar to _pix2pix_ we used a patch discriminator that splits the image in 26 square patches (depending on the image size) and produces a _real_ or _fake_ prediction for each of them.  

In order to try different approaches, we decided to use _Transformers_ in place of the discriminator and the generator.  _Transformers_ are a special architectures of DNN that make extensive use of attention mechanisms [@transformers].  Because of their ability to have larger receptive fields compared to convolutional neural networks (CNNs) that allow tracking long-range dependencies within an image, these attention based architectures have proven very effective in image processing tasks and gave  rise to Visual Transformers or __ViT__ [@ViT].  We tried different architectures with ViTs generators or discriminators and measured a range of metrics for each of them. Finally, we also tried different loss functions to get the best results.
    

## Dataset

For training and validation purposes we used a subset of the COCO dataset of images [@cocodataset] that is provided by the FastAI framework [@fastai]. We downloaded 10.000 images from this dataset and randomly splited them into two sets: a training set with 8.000 images and a validation set with 2.000 images.  Then we resized the images so that they have manageable dimensions that allow feeding into the different network architectures without requiring extremely high computational resources or long times.  Similar to [@pix2pix] data augmentation was achieved by flipping the images horizontally (this is only done for the training set).  We used 16 images on each batch that goes through the network.  Each image was converted to the __Lab__ colorspace and the channels adjusted float values between -1 and 1. 
    

## Loss functions

The loss function of the discriminator for each image is the binary cross entropy between the predictions and the ground truth: _real_ if the real __ab__ channels were fed into the discriminator of _fake_ if the generated __ab__ channels were used instead.  The loss function of the generator was the combination of the L1 error and the loss function of the discriminator as if they were _real_ __ab__ channels.  The intuition behind is that the generator "wins" every time it fools the discriminator into assigning _real_ predictions to the generated outputs.  We used either cross entropy or least squared errors to calculated the loss function of the discriminator.

## Model Training, Transfer and Validation

All the training was done on a machine equipped with an NVIDIA K80 GPU, 4 vCPUs and 61 MB of RAM (_AWS EC2 p2.xlarge_ instance).  The transfer learning, validation and metrics calculation were done in an Intel i5 CPU with 8 MB of RAM. 
For validation we used 2000 images from the __COCO__ dataset that were not used for training, and a set of 70 photographs that were taken by the authors.  To assess the different networks architectures we selected a set of metrics for regression models:

* Correlation coeficient _R_ squared
* Explained variance
* Mean absolute error
* Median absolute error
* Mean squared error

We calculated all these metrics for each one of the __ab__ channels.


```python
class MainModel(nn.Module):
    def __init__(self, net_G=None, net_D=None, use_ViT_gen = False, lr_G=2e-4, lr_D=2e-4, 
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        self.use_ViT_gen = use_ViT_gen
        
        if net_G is None:
            raise NotImplementedError
        else:
            self.net_G = net_G.to(self.device)
        
        if net_D is None:
            self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        else:
            self.net_D = net_D.to(self.device)
            
        #self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)  # Original BCE Loss
        self.GANcriterion = GANLoss(gan_mode='lsgan').to(self.device)  # Final improvement with Least Square Error loss
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
    
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        
    def forward(self):
        if (self.use_ViT_gen == True):
            outputs = self.net_G(self.L.repeat(1,3,1,1))  # Copy the L channel 3 times to mimick the 3-channels input that the pretrained network requires.
            self.fake_color = outputs.logits
        else:
            self.fake_color = self.net_G(self.L)
    
    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
        
    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()
```


## First approach: _ResNet18_ generator

On out first approach we employed a generator based on the ResNet18 network. One of the challenges of GANs is that, at the beginning of the training, the task of the discriminator is much easier than that of the generator because the generated outputs are very different from the real ones.  In this situation, the discriminator learns so much faster and gives no time to the generator to adapt.  To avoid this, we gave the generator a _head start_ by training it alone (without the generator) for 20 epochs with a L1 loss function and saving its weights.  After that we started the parallel training of the generator and the patch discriminator for another 20 epochs.


![Samples of results from using a ResNet18 network as the generator. Top row: inputs, Middle row: predictions, Bottom row: ground truth.](results/ResNet18.png) 

\begin{table}\centering\caption[Metrics of ResNet18 generator on validation dataset]{ResNet18 metrics on validation dataset}\begin{tabular}{llll}\toprule{}                 Metric &              a-channel &             b-channel \\\midrule                R-square &     0.9762 &    0.7944 \\     Explained variance &     0.9763 &    0.8047 \\    Mean absolute error &   0.0221 &   0.0813 \\  Median absolute error &   0.0115 &  0.0533 \\     Mean squared error &  0.0016 &  0.0144 \\            Sample size &                   2000 &                  2000 \\\bottomrule\end{tabular}\end{table}


The results of the model with the ResNet18 generator were acceptable.  However, many times they do not look natural because of an excessive use of colors by the generator that resulted in colorful blotches in the pictures.  In agreement with the visual inspection, the resulting metrics on the validation dataset showed that the network did a pretty good job at predicting the __a__-channel with over 97% of the variance of the channel predicted by the model with a very low mean squared error.  However, the prediction on the __b__-channel was not as good with the model predicting only 80% of its variance.

## Second approach: ViT as discriminator

Since the results obtained from the UNet generator did not look quite natural, our first intention was to improve the discriminator so that it will be better at telling apart the _real_ from the _fake_ images.  To do that we decided to replace the CNN based Patch Discriminator with a Visual Transformer.

![Samples of results from using a ViT as the discriminator network. Top row: inputs, Middle row: predictions, Bottom row: ground truth.](results/ViTasDiscriminator.png)

Unfortunately, the results were not satisfactory.  The discriminator got very good a discriminating real from fake very early on and gave not chance for the generator to adapt.  The final results are just gray images or sepia looking images with almost no color.  This is because the best loss the generator could achieve was by producing an average value on the __ab__ channels disregarding of the inputs.

## Third approach: ViT as generator

Once we understood that to have an truly creative network we should put our efforts on the generator instead of on the discriminator, we decided to include a ViT as the generator. After exploring different options we selected a ViT trained in the task of completing masked images [@Zhenda2021]. Since this model expects a 3-channels input, we mimick just by copying the __L__ channel into each of the input channels.  We replaced the decoder block of this model by 3 convolution layers and activation functions.  These layers converted the 14x14x768 inputs of the encoder into 14x14x512 outputs.  A pixel shuffle layer with an upscale factor of 16 reshaped those outputs into the final 224x224x2 output.  To avoid the problem of the discriminator learning too fast, we kept the pretraining step with the generator alone to give a _head start_ to it.  We trained for 20 epochs.

```python
# Build transformer based generator
# https://huggingface.co/docs/transformers/model_doc/vit

from transformers import ViTForMaskedImageModeling, ViTConfig


def build_VTi_generator():

    ## Pretrained generator ViT, 3-input channels, multilayer decoder.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")

    model.decoder = nn.Sequential(nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1), \
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1), \
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1), \
                                    nn.PixelShuffle(upscale_factor=16))
    model = model.to(device)
    return model

```

![Samples of results from pretraining the ViT generator alone. Top row: inputs, Middle row: predictions, Bottom row: ground truth.](results/Pretrained3channelViT_task_pretraining_result.png)

![Samples of results after training the ViT generator with the discriminator.  Top row: inputs, Middle row: predictions, Bottom row: ground truth.](results/Pretrained3channelViT_task_pretraining_FINAL_result.png)

The results were not very encouraging.  After the pretraining when the generator trained alone, the results were acceptable although the colors were not very bright or varied.  However, when we trained with the discriminator, the discriminator won the game and the generator produced just gray images, as before.

## Fourth approach: LSGAN with ViT

The other main challenge of training a GAN is choosing the right loss function.  During training of regular networks a convergence of the loss function to a small enough value signals that the network has achieved an equilibrium and cannot learn more from the training data.  However, training of a GAN is a two players game in which each one tries to minimize its own loss by maximizing the other player loss.  The generator and discriminator losses should not converge but stay in a permanent unstable equilibrium which signals that the game is still being played.  

The loss function is what gives the gradient the generator needs to learn to fool the discriminator and not all loss functions are equal for this task.  As already mentioned, at the beginning of the training it is very easy for the discriminator to tell fake from real.  When Cross Entropy is used, it can provide very low or vanishing gradients at the start of the training that do not help the improvement of the generator.  To overcome this problem, it has been suggested the use of least squared errors loss functions [@LSGAN]. Therefore, we replaced the BCE loss with least square errors loss to construct a __ViT-LSGAN__.

![Samples of results ViT-LSGAN model. Top row: inputs, Middle row: predictions, Bottom row: ground truth.](results/FINAL_ViT_generator.png)


\begin{table}\centering\caption[Metrics of the ViT-LSGAN model on validation dataset]{ViT metrics on validation dataset}\begin{tabular}{llll}\toprule{}                 Metric &              a-channel &             b-channel \\\midrule               R-square &      0.9856 &     0.8596 \\     Explained variance &     0.9859 &    0.8620 \\    Mean absolute error &   0.0157 &   0.0625 \\Median absolute error &   0.0075 &   0.0374 \\Mean squared error &  0.0010 &  0.0095 \\Sample size &                   2000 &                  2000 \\\bottomrule\end{tabular}\end{table}

The metrics from ViT-LSGAN model were very encouraging.  By using visual transformers we got an improvement over the initial UNet based model on every metric in particular in the __b__-channel which was the most difficult to predict.  For example, this model was able to explain 98% of the variance of the __a__-channel and 86% of the variance of the __b__-channel, against 97% and 80%, respectively, when using the UNet model.

## Image Upscaling

The ViT-LSGAN was trained with 224 by 224 images, so it is better to use downsampled images for transfering color.  To colorize the full size image, we upscaled the outputs of the network by resizing the predicted __ab__ channels to the size of the original __L__ channel, and combined the results with the __L__ channel to produce a color image in the __Lab__ colorspace.


## Final colorization results

We tested our ViT-LSGAN with historic black and white urban pictures as well as with historic portraits.  We noticed that sometimes old photographs are saturated, particularly, in the sky area.  This causes the LSGAN to interpret them as cloudy skies and producing a white sky.  This effect can be partially overcome by adjusting the gain of the original photo before feeding it into the network.
    
![Historic photos of Montevideo.](Project_files/Project_72_0.png)
    
![Historic portraits.](Project_files/Project_73_0.png)
    

# References


