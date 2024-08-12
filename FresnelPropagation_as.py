import torch

def FresnelPropagation_as(input, dx, dy, z, wavelength):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    k = 2 * torch.pi / wavelength
    Nx, Ny = input.shape
    Nxx = 2 * Nx
    Nyy = 2 * Ny

    input2x = torch.zeros(Nxx, Nyy, dtype=input.dtype, device=device)
    start_x = round(Nx/2) - 1
    start_y = round(Ny/2) - 1
    input2x.narrow(0, start_x, Nx).narrow(1, start_y, Ny).copy_(input)

    dal = 1. / (Nxx * dx)  # delta alpha over lambda
    dbl = 1. / (Nyy * dy)  # delta beta over lambda

    al, bl = torch.meshgrid(torch.arange(Nxx, device=device), torch.arange(Nyy, device=device))
    al = (al - Nxx/2) * dal
    bl = (bl - Nyy/2) * dbl
    A = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(input2x)))
    A = A.to(device)
    prop_kernel = torch.exp(1j * 2 * torch.pi * z * torch.sqrt(1 / wavelength ** 2 - al ** 2 - bl ** 2))
    prop_kernel = prop_kernel.to(device)

    # K. Matsushima et al., "Band-limited angular spectrum method for numerical simulation of free-space propagation
    # in far and near fields," Opt. Express 17, 19662 (2009) 참조
    # FresnelPropagationShift_as.m 과 동일. 다만 sx=sy=0
    sx = 0
    sy = 0
    fla = torch.abs(-al * z / torch.sqrt(1 / wavelength ** 2 - al ** 2 - bl ** 2) + sx)
    flb = torch.abs(-bl * z / torch.sqrt(1 / wavelength ** 2 - al ** 2 - bl ** 2) + sy)
    prop_kernel[(fla > 1 / (2 * dal)) | (flb > 1 / (2 * dbl))] = 0


    intermediate = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(A * prop_kernel)))

    output = intermediate.narrow(0, start_x, Nx).narrow(1, start_y, Ny)


    return output