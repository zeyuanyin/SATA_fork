import time
import timm
import argparse
import sata

import torch
import torchvision.datasets as datasets

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# from thop import profile
import tqdm

def evaluate_imagenet(args):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_cuda = True if torch.cuda.is_available() else False
    # Load a pretrained model
    model = timm.create_model(args.model_name, pretrained=True)
    
    print(model.default_cfg["mean"])
    print(model.default_cfg["std"])
    
    
    input_size = model.default_cfg["input_size"][1]
    size_ratio = (256 / 224)
    data_transform = transforms.Compose([
        transforms.Resize(int(size_ratio * input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(model.default_cfg["mean"], model.default_cfg["std"]),
    ])

    # Load ImageNet dataset
    imagenet_dataset = datasets.ImageFolder(args.data_path, transform=data_transform)
    dataloader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Apply sata patch
    # if args.sata:
    sata.patch.timm(model,
                    gamma = args.gamma,
                    alpha = args.alpha)

    # Load the ViT model
    model.to(device)
    model.eval()

    # FLOPs evaluate
    # input_tensor = torch.randn(1, 3, input_size, input_size).cuda()
    # Compute FLOPs
    #flops, params = profile(model, inputs=(input_tensor,))
    #print(f"GFLOPs: {flops/1e9}, Params (M): {params/1e6}")
    
    # Evaluate the model on the entire ImageNet dataset with intermediate performance prints
    top1_correct = 0
    top5_correct = 0
    total = 0
    batch_count = 0
    
    start = time.time()
    with torch.autocast(device.type, enabled=False):
        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(dataloader):
                # print(labels)
                # exit()
                batch_count += 1
    
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
    
                # Top-1 accuracy
                top1_correct += (predicted == labels).sum().item()
    
                # Top-5 accuracy
                _, top5_predicted = outputs.topk(5, 1, largest=True, sorted=True)
                top5_correct += torch.sum(top5_predicted == labels.view(-1, 1)).item()
    
                total += labels.size(0)
    
                if batch_count % 10 == 0:
                    top1_accuracy = top1_correct / total
                    print(f"Intermediate Top-1 Accuracy after {batch_count} batches: {top1_accuracy * 100:.2f}%")

    if is_cuda:
        torch.cuda.synchronize()
    end = time.time()
    elapsed_time = end - start

    throughput = total / elapsed_time

    print(f"Throughput: {throughput:.2f} im/s")
    # Final accuracy
    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total
    print(f"Final Top-1 Accuracy on the entire ImageNet dataset: {top1_accuracy * 100:.2f}%")
    print(f"Final Top-5 Accuracy on the entire ImageNet dataset: {top5_accuracy * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a pretrained ViT model on ImageNet.")
    parser.add_argument("--model_name", type=str, default="deit_base_patch16_224", help="Name of the ViT model")
    parser.add_argument("--alpha", type=float, default=1.0, help="alpha")
    parser.add_argument("--gamma", type=float, default= 0.7, help="gamma")
    parser.add_argument('--sata', default=False, action="store_true", help="use sata")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation")
    parser.add_argument("--data_path", type=str, default="./ImageNet2012/val", help="Path to the ImageNet validation dataset")
    args = parser.parse_args()

    evaluate_imagenet(args)
