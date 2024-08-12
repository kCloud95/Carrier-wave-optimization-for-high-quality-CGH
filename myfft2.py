import torch

def myfft2(input):
    output = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(input)))

    return(output)