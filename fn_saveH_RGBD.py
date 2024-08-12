import torch
import os, sys
sys.path.append('..')
from FresnelPropagation_as import FresnelPropagation_as

def fn_saveHologramRGBD(rgb, depthMap, wavelength, px, py, Nx, Ny, CW):


    allDepth = torch.unique(depthMap)

    hologram = torch.zeros(Ny, Nx)
    hologram = hologram.cuda()
    for idxDepth in range(len(allDepth)):
        z = allDepth[idxDepth]
        layer = rgb * (depthMap == z)
        layer = layer.cuda()
        zo = z

        # carrierPhase = torch.exp(1j * 2 * torch.pi / wavelength * (cu[0] * xx + cu[1] * yy + cu[2] * zo))
        carrierPhase = FresnelPropagation_as(CW, px, py, zo, wavelength)
        layer = layer * carrierPhase

        hologram_temp = FresnelPropagation_as(layer, px, py, -zo, wavelength)

        hologram = hologram + hologram_temp

    # fileName = f'{dirName}/hologram_cuFxIdx{cuFxIdx}_cuFyIdx{cuFyIdx}'
    # torch.save(hologram, fileName)
    return hologram
