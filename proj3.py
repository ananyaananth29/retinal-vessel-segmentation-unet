## Importing Libraries
import os
import gzip
import shutil
import tarfile
import imageio.v2 as imageio
import imagecodecs
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import collections
from skimage import morphology
from skimage.measure import block_reduce
import scipy
from torch.utils.data import Dataset
import PIL
from PIL import Image
from sklearn.metrics import f1_score
import copy
import random
import math
import torch
from packaging import version
import subprocess
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F

##### Checking Torch library requirement #####
my_torch_version = torch.__version__
minimum_torch_version = '1.7'
if version.parse(my_torch_version) < version.parse(minimum_torch_version):
    print('Warning!!! Your Torch version %s does NOT meet the minimum requirement!\
            Please update your Torch library\n' %my_torch_version)

##### Checking the System #####
try:
    hostname = subprocess.getoutput('hostname')
    if 'lab' in hostname[0] and '.eng.utah.edu' in hostname[0]:
        IN_CADE = True
    else:
        IN_CADE = False
except:
    IN_CADE = False

##### Requesting a GPU #####
def define_gpu_to_use(minimum_memory_mb = 3500):    
    thres_memory = 600 
    gpu_to_use = None
    try: 
        os.environ['CUDA_VISIBLE_DEVICES']
        print('GPU already assigned before: ' + str(os.environ['CUDA_VISIBLE_DEVICES']))
        return
    except:
        pass
    
    torch.cuda.empty_cache()
    for i in range(16):
        free_memory = subprocess.getoutput(f'nvidia-smi --query-gpu=memory.free -i {i} --format=csv,nounits,noheader')
        if free_memory[0] == 'No devices were found':
            break
        free_memory = int(free_memory[0])
        
        if free_memory>minimum_memory_mb-thres_memory:
            gpu_to_use = i
            break
            
    if gpu_to_use is None:
        print('Could not find any GPU available with the required free memory of ' + str(minimum_memory_mb) \
              + 'MB. Please use a different system for this assignment.')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_to_use)
        print('Chosen GPU: ' + str(gpu_to_use))
        x = torch.rand((256,1024,minimum_memory_mb-thres_memory)).cuda()
        x = torch.rand((1,1)).cuda()        
        del x

## Request a gpu and reserve the memory space
define_gpu_to_use()

# Part 1 - DRIVE and STARE Dataset 

##### Removing Small Regions #####
## This function removes small regions (<size) of a given binary image.
def remove_small_regions(img, size):
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

##### Resizing Image #####
## This function resizes a given input image to be half of the original size.
def resize_img(img):
    if len(img.shape)==3:
        img = np.array(Image.fromarray(img).resize(((img.shape[1]+1)//2,(img.shape[0]+1)//2), \
                                                   PIL.Image.BILINEAR))
    else:
        img = block_reduce(img, block_size=(2, 2), func=np.max)
    return img

##### DRIVE Dataset Loading Function #####
## This function unzips the downloaded file. Then, it loads the corresponding mask
## for each image in the DRIVE dataset.
def drive_read_images(filetype, dest_folder):
    zip_ref = zipfile.ZipFile('datasets.zip', 'r')
    zip_ref.extractall('datasets/drive')
    zip_ref.close()
    zip_ref = zipfile.ZipFile('datasets/drive/training.zip', 'r')
    zip_ref.extractall('datasets/drive')
    zip_ref.close()
    all_images = []
    for item in sorted(os.listdir(dest_folder)):
        if item.endswith(filetype):
            img = imageio.imread(dest_folder + item)
            if len(img.shape) == 3:
                img = np.pad(img , ((12,12), (69,70),(0,0)), mode = 'constant')
            else:
                img = np.pad(img , ((12,12), (69,70)), mode = 'constant')
            img = resize_img(img)
            img = img/255.
            img = img.astype(np.float32)
            if len(img.shape) == 2:
                img = img.astype(np.float32)
                img = np.expand_dims(img, axis = 2)
            all_images.append(img)
    return all_images

##### STARE Dataset Loading Function #####
## This function untars the downloaded file. Then, it loads the corresponding mask
## for each image in the STARE dataset.
def stare_read_images(tar_filename, dest_folder, do_mask = False):
    tar = tarfile.open(tar_filename)
    tar.extractall(dest_folder)
    tar.close()
    all_images = []
    all_masks = []
    for item in sorted(os.listdir(dest_folder)):
        if item.endswith('gz'):
            with gzip.open(dest_folder + item, 'rb') as f_in:
                with open(dest_folder + item[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(dest_folder + item)
            img = imageio.imread(dest_folder + item[:-3])
            if len(img.shape) == 3:
                img = np.pad(img , ((1,2), (2,2),(0,0)), mode = 'constant')
            else:
                img = np.pad(img , ((1,2), (2,2)), mode = 'constant')
            img = resize_img(img)
            img = img/255.
            img = img.astype(np.float32)
            if len(img.shape) == 2:
                img = img.astype(np.float32)
                img = np.expand_dims(img, axis = 2)
            all_images.append(img)
            if do_mask:
                mask = (1-remove_small_regions(np.prod((img<50/255.)*1.0, axis = 2)>0.5, 1000))*1.0
                mask = np.expand_dims(mask, axis = 2)
                all_masks.append(mask.astype(np.float32))
    if do_mask:
        return all_images, all_masks
    else:
        return all_images

##### Spliting a dataset for training, validatation, and testing #####
## This function splits a given dataset into 3 subsets of 70%-10%-20% for train-val-test,
## respectively, and is used internally in the dataset classes below.
def get_split(array_to_split, split):
    np.random.seed(0)
    np.random.shuffle(array_to_split)
    np.random.seed()
    if split == 'train':
        array_to_split = array_to_split[:int(len(array_to_split)*0.7)]
    elif split == 'val':
        array_to_split = array_to_split[int(len(array_to_split)*0.7):int(len(array_to_split)*0.8)]
    elif split == 'test':
        array_to_split = array_to_split[int(len(array_to_split)*0.8):]
    return array_to_split


# ---------------------------------------- Data Augmentation ----------------------------------------
##### Defining Transformations #####
## The transformations below will be applied to input image, segmentation ground-truth and mask.

## Applying transformations to all array in list x
def _iterate_transforms(transform, x):
    for i, xi in enumerate(x):
        x[i] = transform(x[i])
    return x

## Redefining Pytorch composed transform so that it uses the _iterate_transforms function
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = _iterate_transforms(transform, x)
        return x

## Generating randomize odd for vertical flipping class
class RandomVerticalFlipGenerator(object):
    def __call__(self, img):
        self.random_n = random.uniform(0, 1)
        return img

## Performing vertical flip using randomization provided by generator class above
class RandomVerticalFlip(object):
    def __init__(self, gen):
        self.gen = gen

    def __call__(self, img):
        if self.gen.random_n < 0.5:
            ### Your code starts here ###
            flipped_img = torch.flip(img, dims=[1])  
            ### Your code ends here ###
            return flipped_img
        return img

## Generating randomize odd for horizontal flipping class
class RandomHorizontalFlipGenerator(object):
    def __call__(self, img):
        self.random_n = random.uniform(0, 1)
        return img

## Performing horizontal flip using randomization provided by generator class above
class RandomHorizontalFlip(object):
    def __init__(self, gen):
        self.gen = gen

    def __call__(self, img):
        if self.gen.random_n < 0.5:
            ### Your code starts here ###
            flipped_img = torch.flip(img, dims=[2])
            ### Your code ends here ###
            return flipped_img
        return img


class RetinaDataset(Dataset):
    ##### Initializing the class #####
    def __init__(self, retina_array, split = 'train', do_transform=False):
        ## Split parameter is used to specify which process the data is used for,
        ## and it can be 'train', 'val', and 'test'
        indexes_this_split = get_split(np.arange(len(retina_array), dtype = int), split)
        self.retina_array = [self.transpose_first_index(retina_array[i]) for i in indexes_this_split]
        self.split = split
        self.do_transform = do_transform

    ##### Retrieving a sample with the corresponding index #####
    ## This function retrieve a sample from the dataset at the specified index
    ## and returns a list in which the first element is input image to be segmented,
    ## the second element is the segmentation ground truth, and the last element is the mask of image region
    def __getitem__(self, index):
        sample = [torch.FloatTensor(x) for x in self.retina_array[index]]
        if self.do_transform:
            v_gen = RandomVerticalFlipGenerator()
            h_gen = RandomHorizontalFlipGenerator()
            t = Compose([
                v_gen,
                RandomVerticalFlip(gen=v_gen),
                h_gen,
                RandomHorizontalFlip(gen=h_gen),
            ])
            sample = t(sample)
        return sample

    ##### Accessing the length of the dataset #####
    def __len__(self):
        return len(self.retina_array)

    ##### Flipping the third dimension #####
    def transpose_first_index(self, x):
        x2 = (np.transpose(x[0], [2,0,1]), np.transpose(x[1], [2,0,1]), np.transpose(x[2], [2,0,1]))
        return x2


##### Loading Data #####
## This function loads DRIVE and STARE datasets into a list of list of arrays.
## The first element in the list is the list of input images, the second element is a list of segmentation
## ground truth, and the last element is a list of masks of image region.
## The original images were padded as squares so that we can fit them to a traditional CNN.
## The masks are binary images, and contain the location of the original image (labeled as 1) and
## the padded region (labeled as 0). These masks are used to limit where outputs are backpropagated
## for trained and which region of the image should be used for scoring.
def get_retina_array():
    stare_images, stare_mask = stare_read_images("stare-images.tar", 'datasets/stare/images/', do_mask = True)
    stare_segmentation = stare_read_images("labels-vk.tar", 'datasets/stare/segmentations/')
    drive_training_images = drive_read_images('tif', 'datasets/drive/training/images/')
    #drive_test_images = drive_read_images('tif', 'datasets/drive/test/images/')
    drive_training_segmentation = drive_read_images('gif', 'datasets/drive/training/1st_manual/')
    #drive_test_segmentation = drive_read_images('gif', 'datasets/drive/test/1st_manual/')
    drive_training_mask = drive_read_images('gif', 'datasets/drive/training/mask/')
    #drive_test_mask = drive_read_images('gif', 'datasets/drive/test/mask/')
    return list(zip(stare_images+drive_training_images,#+drive_test_images,
                           stare_segmentation+drive_training_segmentation,#+drive_test_segmentation,
                           stare_mask + drive_training_mask))# + drive_test_mask))

print(f'\nLoading retina data')
retina_array = get_retina_array()

train_dataset = RetinaDataset(retina_array, do_transform = True)
val_dataset = RetinaDataset(retina_array, split = 'val')
test_dataset = RetinaDataset(retina_array, split = 'test')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


## Visualing a few cases in the training set
print('\nVisualing a few cases in the training set')
# Optionally, create a directory to store your saved images
output_dir = "train_plots_ex11"
os.makedirs(output_dir, exist_ok=True)
for batch_idx, (data, segmentation, mask) in enumerate(RetinaDataset(retina_array)):
    if batch_idx % 15 == 0:
        # Save Input Image
        fig1 = plt.figure()
        plt.title("Input Image")
        plt.imshow(data[:,:,:].permute([1,2,0]).cpu().numpy())
        input_filename = os.path.join(output_dir, f"samples_input_image_{batch_idx}.png")
        fig1.savefig(input_filename)
        plt.close(fig1)

        # Save Segmentation Ground Truth
        fig2 = plt.figure()
        plt.title("Segmentation Ground Truth")
        plt.imshow(segmentation[0,:,:].cpu().numpy())
        seg_filename = os.path.join(output_dir, f"samples_segmentation_ground_truth_{batch_idx}.png")
        fig2.savefig(seg_filename)
        plt.close(fig2)

        # Save Mask
        fig3 = plt.figure()
        plt.title("Mask")
        plt.imshow(mask[0,:,:].cpu().numpy())
        mask_filename = os.path.join(output_dir, f"samples_mask_{batch_idx}.png")
        fig3.savefig(mask_filename)
        plt.close(fig3)




##### Calculating F1 metric #####
def get_score_model(model, data_loader):
    ## Toggling model to eval mode
    model.eval()

    ## Turning off gradients
    with torch.no_grad():
        logits_predicted = np.zeros([0, 1, 304, 352])
        segmentations = np.zeros([0, 1, 304, 352])

        mean_f1 = 0.0
        ## Iterating through the dataset and perform inference for each sample.
        ## Then, the F1 score is computed for each sample.
        for image, segmentation, mask  in data_loader:
            image = image.cuda()
            logit_predicted = model(image)
            logit_predicted = logit_predicted.cpu().detach().numpy()*mask.numpy()
            segmentation = segmentation.numpy()*mask.numpy()

            ## Computing F1 score for each sample in the batch
            for i in range(segmentation.shape[0]):
                curr_seg = segmentation[i,...].reshape([-1])
                curr_logit = logit_predicted[i,...].reshape([-1]) > 0
                curr_f1 = f1_score(curr_seg, curr_logit)
                mean_f1 += curr_f1

    ## Returning the mean F1 of the entire dataset
    return mean_f1/len(data_loader.dataset)

# ---------------------------------------- Computing Weight Vector ----------------------------------------
print(f'\nStarting Part 1.2')
# Initialize counters
total_positive = 0
total_negative = 0

# Iterate over dataset to compute total positives and negatives
for data, segmentation, mask in train_loader:
    # Apply mask to segmentation labels to count only relevant pixels
    segmentation_masked = segmentation * mask
    
    # Count positive and negative pixels
    positives = segmentation_masked.sum().item()
    negatives = mask.sum().item() - positives
    
    total_positive += positives
    total_negative += negatives

# Calculate percentage of positive labels
total_labels = total_positive + total_negative
positive_percentage = (total_positive / total_labels) * 100

# Calculate ratio of negative to positive labels
negative_positive_ratio = total_negative / total_positive

# Clearly print the computed values
print(f"Percentage of positive (foreground) labels: {positive_percentage:.2f}%")
print(f"Ratio of negative (background) to positive labels: {negative_positive_ratio:.2f}")


# Define your weighted binary cross-entropy loss function clearly
def weighted_binary_loss(predictions, targets, mask, neg_pos_ratio):
    """
    Weighted Binary Cross Entropy Loss to handle class imbalance.
    
    Args:
        predictions: Predicted logits (raw model outputs before sigmoid).
        targets: Ground truth binary labels.
        mask: Mask to specify valid pixels for loss calculation.
        neg_pos_ratio: Ratio of negative labels to positive labels calculated previously.
        
    Returns:
        loss: Scalar weighted binary cross entropy loss.
    """

    # Apply sigmoid to logits
    predictions = torch.sigmoid(predictions)

    # Compute weighted binary cross entropy loss
    # Define weights tensor: foreground (positive) = neg_pos_ratio, background (negative) = 1
    weights = targets * neg_pos_ratio + (1 - targets)

    # Compute loss clearly using masks and weights
    loss = F.binary_cross_entropy(predictions, targets, weight=weights, reduction='none')

    # Apply the mask to calculate loss only on valid pixels
    masked_loss = loss * mask

    # Compute mean loss over masked pixels
    loss_mean = masked_loss.sum() / mask.sum()

    return loss_mean


# ---------------------------------------- Part 1.3 - Implementing U-Net ----------------------------------------
print(f'\nStarting Part 1.3')

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=16):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")
        
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")
        
        self.conv_final = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv_final(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )

# Instantiate the model clearly
model_ex13 = UNet(in_channels=3, out_channels=1, init_features=16).cuda()
# ---------------------------------------- Training the Model ----------------------------------------

def training_stage(epoch, optimizer, model, train_loader, val_loader):
    model.train()
    loss_values = []
    for batch_idx, (data, segmentation, mask) in enumerate(train_loader):

        # Move data to GPU
        data = data.cuda()
        segmentation = segmentation.cuda()
        mask = mask.cuda()

        # Forward pass
        predictions = model(data)

        # Compute loss using provided mask (you should have defined `negative_positive_ratio` earlier)
        loss_value = weighted_binary_loss(predictions, segmentation, mask, negative_positive_ratio)

        # Backpropagation
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        loss_values.append(loss_value.item())
    return np.mean(loss_values)

optimizer = torch.optim.SGD(model_ex13.parameters(), lr=0.01, momentum=0.9, nesterov=True)
n_epochs = 200

## Using the scheduler module to reduce the learning rate after reaching a plateau.
## More information about the scheduler can be found at
## https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.StepLR
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
current_best_score = -1
print('\nStarting training loop 1')
for epoch in range(n_epochs):
    ## Train the model
    loss_value = training_stage(epoch, optimizer, model_ex13, train_loader, val_loader)
    ## Evaluate the current model
    f1_val = get_score_model(model_ex13, val_loader)
    f1_train = get_score_model(model_ex13, train_loader)
    current_score = f1_val
    ## Save the model
    if current_score > current_best_score:
        current_best_score = current_score
        best_model_ex13 = copy.deepcopy(model_ex13)
    print('  Train Epoch: {:d} \tLoss: {:.5f}'.format(epoch,loss_value))
    print('  F1 train: {:.5f} \t F1 val: {:.5f}'.format(f1_train, f1_val))
    print('    Best F1 so far: {:.5f}'.format(current_best_score))
    ## Activate scheduler
    scheduler.step()
print('    F1 score of test set: {:.5f}'.format(get_score_model(best_model_ex13, test_loader)))


# ---------------------------------------- Visualizing Validation Predictions ----------------------------------------
output_dir = "val_predictions_ex13"
os.makedirs(output_dir, exist_ok=True)

best_model_ex13.eval()

with torch.no_grad():
    for idx, (data, segmentation, mask) in enumerate(val_loader):
        if idx >= 5:  # Limit to 5 examples for clarity
            break

        # Move to GPU
        data = data.cuda()
        segmentation = segmentation.cuda()
        mask = mask.cuda()

        # Predict
        prediction = best_model_ex13(data)
        prediction = torch.sigmoid(prediction)  # convert logits to probabilities
        prediction_binary = (prediction > 0.5).float()

        # Convert for plotting
        img_np = data[0].cpu().permute(1, 2, 0).numpy()
        gt_np = segmentation[0][0].cpu().numpy()
        pred_np = prediction_binary[0][0].cpu().numpy()

        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img_np)
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        axs[1].imshow(gt_np, cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

        axs[2].imshow(pred_np, cmap='gray')
        axs[2].set_title("Predicted Mask")
        axs[2].axis("off")

        fig.suptitle(f"Validation Sample {idx}")
        fig.tight_layout()
        fig.savefig(f"{output_dir}/val_sample_{idx}.png", dpi=300)
        plt.close(fig)

        # Print analysis snippet to include in report
        false_positives = ((pred_np == 1) & (gt_np == 0)).sum()
        false_negatives = ((pred_np == 0) & (gt_np == 1)).sum()
        print(f"Sample {idx}: False Positives = {false_positives}, False Negatives = {false_negatives}")


# ---------------------------------------- Part 1.4 - Implementing U-Net Without Skip Connections ----------------------------------------

class UNetNoSkip(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=16):
        super(UNetNoSkip, self).__init__()

        features = init_features
        self.encoder1 = UNetNoSkip._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = UNetNoSkip._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = UNetNoSkip._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = UNetNoSkip._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNetNoSkip._block(features * 8, features * 16, name="bottleneck")

        # Upsample with doubled output channels (no skip concat)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 16, kernel_size=2, stride=2)
        self.decoder4 = UNetNoSkip._block(features * 16, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 8, kernel_size=2, stride=2)
        self.decoder3 = UNetNoSkip._block(features * 8, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 4, kernel_size=2, stride=2)
        self.decoder2 = UNetNoSkip._block(features * 4, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features * 2, kernel_size=2, stride=2)
        self.decoder1 = UNetNoSkip._block(features * 2, features, name="dec1")

        self.conv_final = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(dec1)

        return self.conv_final(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )

# Instantiate the model
model_ex15 = UNetNoSkip(in_channels=3, out_channels=1, init_features=16).cuda()

optimizer = torch.optim.SGD(model_ex15.parameters(), lr=0.01, momentum=0.9, nesterov=True)
n_epochs = 200
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

current_best_score = -1
print('\nStarting training loop 2')
for epoch in range(n_epochs):
    ## Train the model
    loss_value = training_stage(epoch, optimizer, model_ex15, train_loader, val_loader)
    f1_val = get_score_model(model_ex15, val_loader)
    f1_train = get_score_model(model_ex15, train_loader)
    current_score = f1_val
    ## Save the model
    if current_score > current_best_score:
        current_best_score = current_score
        best_model_ex15 = copy.deepcopy(model_ex15)
    print('  Train Epoch: {:d} \tLoss: {:.5f}'.format(epoch,loss_value))
    print('  F1 train: {:.5f} \t F1 val: {:.5f}'.format(f1_train, f1_val))
    ## Activate scheduler
    scheduler.step()

print('    F1 score of test set: {:.5f}'.format(get_score_model(best_model_ex15, test_loader)))

output_dir = "val_predictions_ex15"
os.makedirs(output_dir, exist_ok=True)

best_model_ex15.eval()

with torch.no_grad():
    for idx, (data, segmentation, mask) in enumerate(val_loader):
        if idx >= 5:  # Show 5 samples max
            break

        data = data.cuda()
        segmentation = segmentation.cuda()
        mask = mask.cuda()

        prediction = best_model_ex15(data)
        prediction = torch.sigmoid(prediction)
        prediction_binary = (prediction > 0.5).float()

        # Convert tensors to NumPy for plotting
        img_np = data[0].cpu().permute(1, 2, 0).numpy()
        gt_np = segmentation[0][0].cpu().numpy()
        pred_np = prediction_binary[0][0].cpu().numpy()

        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img_np)
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        axs[1].imshow(gt_np, cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

        axs[2].imshow(pred_np, cmap='gray')
        axs[2].set_title("Predicted Mask (No Skip)")
        axs[2].axis("off")

        fig.suptitle(f"Validation Sample {idx} - model_ex15")
        fig.tight_layout()
        fig.savefig(f"{output_dir}/val_sample_{idx}.png", dpi=300)
        plt.close(fig)

        # Print for qualitative analysis
        false_positives = ((pred_np == 1) & (gt_np == 0)).sum()
        false_negatives = ((pred_np == 0) & (gt_np == 1)).sum()
        print(f"\n🩺 [Validation Sample {idx}]")
        print(f"Model: U-Net Without Skip Connections")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")


print('\nCOMPLETED!')