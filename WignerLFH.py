import numpy as np
import torch
import time

class WignerLFH:
    def LFtoH(self, FTLF, W, NxL,NyL, Nu, dxL, dxH):

        tic = time.time()

        self.FTLF = FTLF
        self.W = W
        self.NxL = NxL
        self.NyL = NyL
        self.Nu = Nu
        self.Nv = Nu
        self.dxL = dxL
        self.dyL = dxL
        self.dxH = dxH

        M = self.dxL/self.dxH

        bufferX = np.floor(self.Nu/2)+1
        bufferY = np.floor(self.Nv/2)+1

        NxH_NB = np.round(M*self.NxL)
        NyH_NB = np.round(M*self.NyL)

        
        
        NxH = NxH_NB + bufferX*2
        self.NxH = NxH
        NyH = NyH_NB + bufferY*2
        self.NyH = NyH

        #hologram = torch.zeros_like(W)
        hologram = torch.complex(torch.zeros((int(NyH),int(NxH))),torch.zeros((int(NyH),int(NxH)))) #.cuda()
        # hologram = hologram.cuda()
        
        for idxTauX in range(1,self.Nu):

            startXH = int(np.floor( (idxTauX-1)/2 ) + np.floor(- (self.Nu+1)/4 - NxH_NB/2 + self.NxH/2))
            endXH = int(startXH + NxH_NB )
            startXrpm = int(np.floor(-(idxTauX-1)/2 ) + np.floor( (self.Nu+1)/4 - NxH_NB/2 + self.NxH/2))
            endXrpm = int(startXrpm + NxH_NB )

            for idxTauY in range(1,self.Nv):

                startYH = int(np.floor( (idxTauY-1)/2 ) + np.floor( - (self.Nv+1)/4 - NyH_NB/2 + self.NyH/2))
                endYH = int(startYH + NyH_NB )
                startYrpm = int(np.floor(-(idxTauY-1)/2 ) + np.floor( (self.Nv+1)/4 - NyH_NB/2 + self.NyH/2))
                endYrpm = int(startYrpm + NyH_NB )
                # common = FTLF[:, :, idxTauY, idxTauX].cuda()
                FTLF_real = FTLF[:, :, idxTauY, idxTauX].real
                FTLF_imag = FTLF[:, :, idxTauY, idxTauX].imag
                common_real = torch.nn.functional.interpolate(
                FTLF_real.unsqueeze(0).unsqueeze(0),
                size=(int(NyH_NB),int(NxH_NB)),
                mode='bilinear',
                align_corners=True,
                ).squeeze() #.cuda()
                common_imag = torch.nn.functional.interpolate(
                FTLF_imag.unsqueeze(0).unsqueeze(0),
                size=(int(NyH_NB),int(NxH_NB)),
                mode='bilinear',
                align_corners=True,
                ).squeeze() #.cuda()
                common = common_real + 1j*common_imag
                del common_real
                del common_imag
                del FTLF_imag
                del FTLF_real
                common = common.cpu().detach()
                
                hologram[startYH:endYH, startXH:endXH] = hologram[startYH:endYH, startXH:endXH] + common*self.W[startYrpm:endYrpm, startXrpm:endXrpm]
                del common


        toc = time.time()
        print('Processing time for synthesize hologram %f' %(toc-tic))    
        return(hologram)
        


