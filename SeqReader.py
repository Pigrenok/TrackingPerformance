import numpy as np
import torch
import torch.cuda

from matplotlib import pyplot as plt
from matplotlib import gridspec as grs

from os import path

import struct

class seqFileHandler:
    def __init__(self,fileName):
        '''
        This class constructor initialtes seqFileHandler class.

        Input:
            - `fileName`: string, with the path to the StreamPix Sequence file (it can operate with v.4 and v.5 file formats).
        '''
        self._f = open(file=fileName,mode='rb')
        direc = path.dirname(fileName)
        file = path.basename(fileName)
        filen,extn = path.splitext(file)
        self._modFilePath = direc+'/'+filen+'_mod'+extn
        self._filename = filen
        self._extension = extn
        self._directory = direc

        self._header = self._readSeqHeader(self._f)
        [self._version,self._description,self.imageWidth,self.imageHeight,self.imageBitDepth,\
         self.imageSize,\
         self._imageFormat,self.frameNum,self._trueImageSize,self.frameRate,self._compressionFormat,self.refTime,\
         self.refTimeMS,self.refTimeUS] = self._convertHeader(self._header)
        self._trueImageSize = self._trueImageSize[0]
        self._compressionFormat = self._compressionFormat[0]
        self.frameNum = self.frameNum[0]
        if (self._compressionFormat!=0):
            raise ValueError('This tool can handle only uncompressed sequences!')
        if (self._imageFormat!=100):
            raise ValueError('This tool can handle only Monochrome Image (LSB) format!')
        self._shift = 1024
        if type(self._version) is tuple:
            self._version = self._version[0]
        if (self._version == 5):
            self._shift = 8192
        self._header[2]=struct.pack('<I',5)

        self.currentFrame = torch.ShortTensor(np.zeros((self.imageHeight,self.imageWidth),dtype=np.int16))

        self.currFrameNum = 0

        self._currTime = self.refTime
        self._currTimeMS = self.refTimeMS
        self._currTimeUS = self.refTimeUS

    def __del__(self):
        self._f.close()

    def _readSeqHeader(self,fileHandle):
        addr = [0,4,28,32,36,548,572,576,580,584,592,596,600,604,608,612,616,620,624,628,630,632,636,640,644,648,656,660,664]
        length = [4,24,4,4,512,24,4,4,4,8,4,4,4,4,4,4,4,4,4,2,2,4,4,4,4,8,4,4,360]
        headerData = []
        for i in range(29):
            headerData.append(fileHandle.read(length[i]))
        return headerData

    def _convertHeader(self,headerData):
        version = struct.unpack('<L',headerData[2])
        description = headerData[4].replace(b'\x00',b'')
        imageWidth,imageHeight,imageBitDepth,imageBitDepthReal,imageSize,imageFormat = struct.unpack('<LLLLLL',headerData[5])
        frameNum = struct.unpack('<L',headerData[6])
        trueImageSize = struct.unpack('<L',headerData[8])
        frameRate = struct.unpack('<d',headerData[9])
        compressionFormat = struct.unpack('<L',headerData[17])
        refTime = struct.unpack('<L',headerData[18])
        refTimeMS = struct.unpack('<H',headerData[19])
        refTimeUS = struct.unpack('<H',headerData[20])
        return version,description,imageWidth,imageHeight,imageBitDepth,imageSize,imageFormat,frameNum,\
    trueImageSize,frameRate,compressionFormat,refTime,refTimeMS,refTimeUS

    def readFrame(self,frame=None):
        if frame==None:
            self.currFrameNum +=1
            self._f.seek(self._shift+(self.currFrameNum-1)*self._trueImageSize)
            dat = self._f.read(self.imageSize)
            self.currentFrame = torch.ShortTensor(np.reshape(np.array(list(dat),dtype=np.int16),(self.imageHeight,self.imageWidth)))
            self._currTime,self._currTimeMS,self._currTimeUS = struct.unpack('<LHH',self._f.read(8))
        if type(frame) == int:
            self._f.seek(self._shift+(frame-1)*self._trueImageSize)
            dat = self._f.read(self.imageSize)

            return torch.ShortTensor(np.reshape(np.array(list(dat),dtype=np.uint8),(self.imageHeight,self.imageWidth)))

    def readFrames(self,startFrame=None,endFrame=None):

        '''
            Function readFrames returns stacked 3-dimensional array of greyscale images extracted from the sequence. The function will extract all consecutive frames from startFrame to endFrame (inclusive).

            Inputs:
                - `startFrame`: None or number type. Number of start frame (inclusive) for the sequence of frames to extract. If None, will be used frame next after current frame recorded.
                - `endFrame`: None or number type. Number of end frame (inclusive) for the sequence of frames to extract. If None, will be used the last frame of the openned sequence.
            Outputs:
                - 3-dimensional torch.ShortTensor with shape BxHxW, where H - height of the frame, W - width of the fame, B - batch dimension, or number of frame in extracted sequence.
        '''
        if startFrame is None:
            startFrame=self.curFrameNum+1
        else:
            try:
                startFrame = torch.tensor(startFrame,dtype=torch.int)
                startFrame = torch.clamp(startFrame,1,self.frameNum)
            except ValueError:
                raise ValueError(f'`startFrame` can be None or numeric, but {type(startFrame)}')

        if endFrame is None:
            endFrame=self.frameNum
        else:
            try:
                endFrame = torch.tensor(endFrame,dtype=torch.int)
                endFrame = torch.clamp(endFrame,1,self.frameNum)
            except ValueError:
                raise ValueError(f'`endFrame` can be None or numeric, but {type(endFrame)}')

        imageList = []
        for frameNum in range(startFrame,endFrame+1):
            imageList.append(self.readFrame(frameNum).unsqueeze(dim=0))
        return torch.cat(imageList,dim=0)


    def gotoFrame(self,frame=0):
        self.currFrameNum=frame
        if frame==0:
            self.currentFrame = torch.ShortTensor(np.zeros((self.imageHeight,self.imageWidth),dtype=np.int16))
        else:
            self.currentFrame = self.readFrame(frame)

    def calculateNoise(self,startFrame,numToAnalyse=100):
        resList = []
        for i in range(startFrame,startFrame+numToAnalyse):
            resList.append(self.readFrame(i))
        if torch.cuda.is_available():
            res = torch.stack(resList).cuda()
        else:
            res = torch.stack(resList)
        del(resList)
        meanFrame = torch.mean(res.float(),0)
        stdFrame = torch.std(res.float(),0)
        rangeFrame = torch.max(res.float(),0)[0] - torch.min(res.float(),0)[0]
        return meanFrame.cpu(),stdFrame.cpu(),rangeFrame.cpu()

    def plotNoiseData(self,startFrame,numToAnalyse=100,titleAddition=None,
                      save=True,saveFolder=None,doScatter=True,doHeatMap=True,
                      scatterXMax=None,scatterSTDYMax=None,scatterRangeYMax=None,xHistMax=None,yHistMax=None):
        if save:
            if saveFolder:
                _saveFolder = saveFolder
            else:
                _savefolder = self._directory

        meanFrame,stdFrame,rangeFrame = self.calculateNoise(startFrame,numToAnalyse)
        if titleAddition:
            titleSTD = 'Standard deviation - {}'.format(titleAddition)
            _filenameSTD = 'std_{}.png'.format(titleAddition)
        else:
            titleSTD = 'Standard deviation - Frames {} - {}'.format(startFrame,startFrame+numToAnalyse)
            _filenameSTD = 'std_frames_{}-{}.png'.format(startFrame,startFrame+numToAnalyse)
        _filenameSTD = _filenameSTD.replace(' ','_')
        if titleAddition:
            titleRange = 'Range - {}'.format(titleAddition)
            _filenameRange = 'range_{}.png'.format(titleAddition)
        else:
            titleRange = 'Range - Frames {} - {}'.format(startFrame,startFrame+numToAnalyse)
            _filenameRange = 'range_frames_{}-{}.png'.format(startFrame,startFrame+numToAnalyse)
        _filenameRange = _filenameRange.replace(' ','_')

        if doScatter:
            meanMax = torch.max(meanFrame)
            stdMax = torch.max(stdFrame)
            rangeMax = torch.max(rangeFrame)

            fig = plt.figure(figsize=(30,20))
            grs.GridSpec(4,4)
            ax1 = plt.subplot2grid((4,4),(0,1),colspan=3,rowspan=3)#fig.add_subplot(222)
            ax1.scatter(meanFrame.view(1,-1),stdFrame.view(1,-1))
            ax1.tick_params('y',direction='in',labelleft=True,pad=-22)
            ax1.tick_params('x',direction='in',labelbottom=True,pad=-18)
            ax1.set_xlim(left=-0.04*meanMax,right=scatterXMax)
            ax1.set_ylim(bottom=-0.5,top=scatterSTDYMax)
            histDataB,binsB = np.histogram(stdFrame.view(1,-1),bins=np.int(torch.ceil(stdMax*4)))
            ax2 = plt.subplot2grid((4,4),(0,0),rowspan=3,xscale='log')#,yticks=[]) #fig.add_subplot(221,xscale='log',yticks=[])
            ax2.barh((binsB[1:]+binsB[:-1])/2,histDataB,align='center')
            ax2.invert_xaxis()
            ax2.set_ylim(bottom=-0.5,top=scatterSTDYMax)
            ax2.set_xlim(right=0.001,left=yHistMax)
            ax2.tick_params('y',direction='in',labelleft=True,pad=-22)
            histDataA,binsA = np.histogram(meanFrame.view(1,-1),bins=15)
            ax3 = plt.subplot2grid((4,4),(3,1),colspan=3,yscale='log')#,xticks=[]) #fig.add_subplot(224,yscale='log',xticks=[])
            ax3.bar((binsA[1:]+binsA[:-1])/2,histDataA,width=15,align='center')
            ax3.invert_yaxis()
            ax3.set_xlim(left=-0.04*meanMax,right=scatterXMax)
            ax3.set_ylim(top=0.001,bottom=xHistMax)
            ax3.tick_params('x',direction='in',labelbottom=True,pad=-18)
            plt.subplots_adjust(top=0.95,wspace=0,hspace=0)
            fig.suptitle(titleSTD)
            if save:
                plt.savefig(_savefolder+'/scatter_'+_filenameSTD)
            else:
                plt.show()
            fig2 = plt.figure(figsize=(30,20))
            grs.GridSpec(4,4)
            ax4 = plt.subplot2grid((4,4),(0,1),colspan=3,rowspan=3)#fig.add_subplot(222)
            ax4.scatter(meanFrame.view(1,-1),rangeFrame.view(1,-1))
            ax4.tick_params('y',direction='in',labelleft=True,pad=-22)
            ax4.tick_params('x',direction='in',labelbottom=True,pad=-18)
            ax4.set_xlim(left=-0.04*meanMax,right=scatterXMax)
            ax4.set_ylim(bottom=-0.5,top=scatterRangeYMax)
            histDataB,binsB = np.histogram(rangeFrame.view(1,-1),bins=np.int(torch.ceil(rangeMax*2)))
            ax5 = plt.subplot2grid((4,4),(0,0),rowspan=3,xscale='log')#,yticks=[]) #fig.add_subplot(221,xscale='log',yticks=[])
            ax5.barh((binsB[1:]+binsB[:-1])/2,histDataB,align='center')
            ax5.invert_xaxis()
            ax5.set_ylim(bottom=-0.5,top=scatterRangeYMax)
            ax5.set_xlim(right=0.001,left=yHistMax)
            ax5.tick_params('y',direction='in',labelleft=True,pad=-22)
            histDataA,binsA = np.histogram(meanFrame.view(1,-1),bins=15)
            ax6 = plt.subplot2grid((4,4),(3,1),colspan=3,yscale='log')#,xticks=[]) #fig.add_subplot(224,yscale='log',xticks=[])
            ax6.bar((binsA[1:]+binsA[:-1])/2,histDataA,width=15,align='center')
            ax6.invert_yaxis()
            ax6.set_xlim(left=-0.04*meanMax,right=scatterXMax)
            ax6.set_ylim(top=0.001,bottom=xHistMax)
            ax6.tick_params('x',direction='in',labelbottom=True,pad=-18)
            plt.subplots_adjust(top=0.95,wspace=0,hspace=0)
            fig.suptitle(titleRange)
            if save:
                plt.savefig(_savefolder+'/scatter_'+_filenameRange)
            else:
                plt.show()


        if doHeatMap:
            fig3 = plt.figure(figsize=(16,20))
            plt.imshow(meanFrame,cmap='gray')
            hm = plt.imshow(stdFrame,cmap='jet',alpha=0.8)
            plt.colorbar(hm)
            plt.title(titleSTD)
            if save:
                plt.savefig(_savefolder+'/heatmap_'+_filenameSTD)
            else:
                plt.show()
            fig4 = plt.figure(figsize=(16,20))
            plt.scatter(meanFrame.view(1,-1),rangeFrame.view(1,-1))
            plt.imshow(meanFrame,cmap='gray')
            hm = plt.imshow(rangeFrame,cmap='jet',alpha=0.8)
            plt.colorbar(hm)
            plt.title(titleRange)
            if save:
                plt.savefig(_savefolder+'/heatmap_'+_filenameRange)
            else:
                plt.show()

            plt.figure(figsize=(16,20))



            plt.show()

    def _writeHeader(self,numFrames,fr):
        for i in range(len(self._header)):
            if i ==6:
                print(numFrames)
                fr.write(struct.pack('<I',numFrames))
            else:
                fr.write(self._header[i])
        fr.write(bytearray(8192-1024))

    def writeAllFrames(self,start=1,numF='all'):
        if numF == 'all':
            numFrames = self.frameNum
        else:
            if type(numF) == int:
                numFrames = numF
            else:
                raise ValueError("Wrong format of frame number!")
        print('Creating file at {0}'.format(self._modFilePath) )
        fr = open(self._modFilePath,mode='wb')
        self._writeHeader(numFrames,fr)
        self.gotoFrame(start)
        print('Reading sequence and writing it to a new file')
        print('Progress: 0%',end='')

        for i in range(1,numFrames+1):
            self.readFrame()
            fr.write(bytes(self.currentFrame.view((1,-1)).byte().numpy()))
            fr.write(struct.pack('<LHH',self._currTime,self._currTimeMS,self._currTimeUS))
            fr.write(bytearray(self._trueImageSize-self.imageHeight*self.imageWidth-8))
            print('\rProgress: %0.2f%%' % (i*100/numFrames),end='')
        print('\rProgress: 100.00%           \n')
        fr.close()


    def writeDiffFrames(self,start=1,numF='all',averaging='doubleEx',ith=5,alpha=0.3,beta=0.3,threshold=2,fileName=None,directory=None):
        if numF == 'all':
            numFrames = self.frameNum
        else:
            if type(numF) == int:
                numFrames = numF
            else:
                raise ValueError("Wrong format of frame number!")
        averagingFound=False
        averagingID=2
        if averaging=='doubleEx':
            averagingFound==True
            averagingID=2
        if ((not averagingFound) and averaging=='singleEx'):
            averagingFound==True
            averagingID=1
        if ((not averagingFound) and averaging=='ith'):
            averagingFound==True
            averagingID=0
        curDir=self._directory
        if directory:
            curDir = directory
        curFile=self._filename+'_mod.'+self._extension
        if fileName:
            curFile = fileName
        print('Creating file at {0}'.format(curDir+'/'+curFile))
        fr = open(curDir+'/'+curFile,mode='wb')

        self._writeHeader(numFrames,fr)
        self.gotoFrame(start)

        print('Preparing Moving Average Tensors...')
        if torch.cuda.is_available():
            print('CUDA device found. Tensors are created in GPU memory.')
            if averagingID==0:
                ithFrames = []
                for i in range(1,ith+1):
                    ithFrames.append(self.currentFrame.view(1,-1).cuda())
                    self.readFrame()
            if averagingID==1:
                movingAverage = self.currentFrame.view(1,-1).cuda()
            if averagingID==2:
                levelAverage = self.currentFrame.view(1,-1).cuda()
                trendAverage = self.currentFrame.view(1,-1).cuda()#torch.cuda.ShortTensor(self.imageWidth*self.imageHeight)
        else:
            print('CUDA device not found. Tensors are created in RAM.')
            if averagingID==0:
                ithFrames = []
                for i in range(1,ith+1):
                    ithFrames.append(self.currentFrame.view(1,-1))
                    self.readFrame()
            if averagingID==1:
                movingAverage = self.currentFrame.view(1,-1)
            if averagingID==2:
                levelAverage = self.currentFrame.view(1,-1)
                trendAverage = self.currentFrame.view(1,-1).cuda()#torch.ShortTensor(self.imageWidth*self.imageHeight)



        print('Reading sequence, finding difference and writing it to a new file')
        print('Progress: 0%',end='')
        startFrame = 1
        if averagingID==0:
            startFrame = ith+1
        for i in range(startFrame+1,numFrames+1):
            self.readFrame()
            if torch.cuda.is_available():
                currFrame = self.currentFrame.view(1,-1).cuda()
            else:
                currFrame = self.currentFrame.view(1,-1)

            if averagingID==0:
                diff = ((ithFrames[0]-currFrame).float()).short()
                ithFrames.remove(ithFrames[0])
                ithFrames.append(currFrame)

            if averagingID==1:
                diff = (torch.abs((movingAverage-currFrame).float())).short()
                movingAverage = torch.round(alpha*currFrame.float()+(1-alpha)*movingAverage.float()).short()

            if averagingID==2:
                movingAverage = levelAverage+trendAverage
                diff = (torch.abs((movingAverage-currFrame).float())).short()
                newLevelAverage = torch.round(alpha*currFrame.float() + (1-alpha)*(movingAverage.float())).short()
                trendAverage = torch.round(beta*(newLevelAverage.float()-levelAverage.float())+(1-beta)*trendAverage.float()).short()
                levelAverage = newLevelAverage

            diff[diff<=threshold]=0
            diff[diff>threshold]=255

            if torch.cuda.is_available():
                frameToWrite = diff.cpu()
            else:
                frameToWrite = diff

            fr.write(bytes(frameToWrite.byte().numpy()))
            fr.write(struct.pack('<LHH',self._currTime,self._currTimeMS,self._currTimeUS))
            fr.write(bytearray(self._trueImageSize-self.imageHeight*self.imageWidth-8))
            print('\rProgress: %0.2f%%' % (i*100/numFrames),end='')
        print('\rProgress: 100.00%           \n')
        fr.close()