import ctypes as ct
from numpy.ctypeslib import ndpointer
import numpy as np
import os
import sys

class TPeakPar(ct.Structure):
    _fields_ = [("label", ct.c_char*64),
                ("mass", ct.c_float),
                ("loMass", ct.c_float),
                ("hiMass", ct.c_float)]

libname = {'linux':'libtwh5.so', 'linux2':'libtwh5.so', 'darwin':'libtwh5.dylib', 'win32':'TwH5Dll.dll'}
h5lib = ct.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), libname[sys.platform]))

class TwH5Desc(ct.Structure):
    _fields_ = [("nbrSamples", ct.c_int),
                ("nbrPeaks", ct.c_int),
                ("nbrWaveforms", ct.c_int),
                ("nbrSegments", ct.c_int),
                ("nbrBlocks", ct.c_int),
                ("nbrMemories", ct.c_int),
                ("nbrBufs", ct.c_int),
                ("nbrWrites", ct.c_int),
                ("nbrLogEntries", ct.c_int),
                ("secondTof", ct.c_int8),
                ("hasSumSpectrum", ct.c_int8),
                ("hasSumSpectrum2", ct.c_int8),
                ("hasBufTimes", ct.c_int8),
                ("hasTofData", ct.c_int8),
                ("hasTofData2", ct.c_int8),
                ("hasPeakData", ct.c_int8),
                ("hasPeakData2", ct.c_int8),
                ("hasTpsData", ct.c_int8),
                ("hasNbrMemories", ct.c_int8),
                ("hasPressureData", ct.c_int8),
                ("hasLogData", ct.c_int8),
                ("hasMassCalibData", ct.c_int8),
                ("hasMassCalib2Data", ct.c_int8),
                ("hasCh1RawData", ct.c_int8),
                ("hasCh2RawData", ct.c_int8),
                ("hasRawDataDesc", ct.c_int8),
                ("hasEventList", ct.c_int8),
                ("alignmentDummy", ct.c_int8),
                ("alignmentDummy2", ct.c_int8),
                ("eventListMaxElementLength", ct.c_int),
                ("daqMode", ct.c_int),
                ("acquisitionMode", ct.c_int),
                ("massCalibMode", ct.c_int),
                ("massCalibMode2", ct.c_int),
                ("nbrCalibParams", ct.c_int),
                ("nbrCalibParams2", ct.c_int),
                ("p", ct.c_double*16),
                ("p2", ct.c_double*16),
                ("tofPeriod", ct.c_double),
                ("blockPeriod", ct.c_double),
                ("sampleInterval", ct.c_float),
                ("singleIonSignal", ct.c_float),
                ("singleIonSignal2", ct.c_float)
               ]

_double_array = ndpointer(dtype=np.double, flags='CONTIGUOUS')
_float_array = ndpointer(dtype=np.float32, flags='CONTIGUOUS')

#replacement for TwRetVal enum
TwDaqRecNotRunning      = 0
TwAcquisitionActive     = 1
TwNoActiveAcquisition   = 2
TwFileNotFound          = 3
TwSuccess               = 4
TwError                 = 5
TwOutOfBounds           = 6
TwNoData                = 7
TwTimeout               = 8
TwValueAdjusted         = 9
TwInvalidParameter      = 10
TwInvalidValue          = 11
TwAborted               = 12


geth5descriptor = h5lib.TwGetH5Descriptor if os.name=='posix' else h5lib._TwGetH5Descriptor
def TwGetH5Descriptor(filename, h5Descriptor):
    geth5descriptor.argtypes = [ct.c_char_p, ct.POINTER(TwH5Desc)]
    return geth5descriptor(filename, ct.pointer(h5Descriptor))

closeh5 = h5lib.TwCloseH5 if os.name=='posix' else h5lib._TwCloseH5
def TwCloseH5(filename):
    closeh5.argtypes = [ct.c_char_p]
    return closeh5(filename)

closeall = h5lib.TwCloseAll if os.name=='posix' else h5lib._TwCloseAll
def TwCloseAll():
    return closeall()

getsumspectrumfromh5 = h5lib.TwGetSumSpectrumFromH5 if os.name=='posix' else h5lib._TwGetSumSpectrumFromH5
def TwGetSumSpectrumFromH5(filename, spectrum, normalize):
    getsumspectrumfromh5.argtypes = [ct.c_char_p, _double_array, ct.c_int8]
    return getsumspectrumfromh5(filename, spectrum, normalize)

getsumspectrum2fromh5 = h5lib.TwGetSumSpectrum2FromH5 if os.name=='posix' else h5lib._TwGetSumSpectrum2FromH5
def TwGetSumSpectrum2FromH5(filename, spectrum, normalize):
    getsumspectrum2fromh5.argtypes = [ct.c_char_p, _double_array, ct.c_int8]
    return getsumspectrum2fromh5(filename, spectrum, normalize)

gettofspectrumfromh5 = h5lib.TwGetTofSpectrumFromH5 if os.name=='posix' else h5lib._TwGetTofSpectrumFromH5
def TwGetTofSpectrumFromH5(filename, spectrum, segmentIndex, segmentEndIndex, bufIndex, bufEndIndex, writeIndex, writeEndIndex, bufWriteLinked, normalize):
    gettofspectrumfromh5.argtypes = [ct.c_char_p, _float_array, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int8, ct.c_int8]
    return gettofspectrumfromh5(filename, spectrum, segmentIndex, segmentEndIndex, bufIndex, bufEndIndex, writeIndex, writeEndIndex, bufWriteLinked, normalize)

gettofspectrum2fromh5 = h5lib.TwGetTofSpectrum2FromH5 if os.name=='posix' else h5lib._TwGetTofSpectrum2FromH5
def TwGetTofSpectrum2FromH5(filename, spectrum, segmentIndex, segmentEndIndex, bufIndex, bufEndIndex, writeIndex, writeEndIndex, bufWriteLinked, normalize):
    gettofspectrum2fromh5.argtypes = [ct.c_char_p, _float_array, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int8, ct.c_int8]
    return gettofspectrum2fromh5(filename, spectrum, segmentIndex, segmentEndIndex, bufIndex, bufEndIndex, writeIndex, writeEndIndex, bufWriteLinked, normalize)

getstickspectrumfromh5 = h5lib.TwGetStickSpectrumFromH5 if os.name=='posix' else h5lib._TwGetStickSpectrumFromH5
def TwGetStickSpectrumFromH5(filename, spectrum, segmentIndex, segmentEndIndex, bufIndex, bufEndIndex, writeIndex, writeEndIndex, bufWriteLinked, normalize):
    getstickspectrumfromh5.argtypes = [ct.c_char_p, _float_array, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int8, ct.c_int8]
    return getstickspectrumfromh5(filename, spectrum, segmentIndex, segmentEndIndex, bufIndex, bufEndIndex, writeIndex, writeEndIndex, bufWriteLinked, normalize)

getstickspectrum2fromh5 = h5lib.TwGetStickSpectrum2FromH5 if os.name=='posix' else h5lib._TwGetStickSpectrum2FromH5
def TwGetStickSpectrum2FromH5(filename, spectrum, segmentIndex, segmentEndIndex, bufIndex, bufEndIndex, writeIndex, writeEndIndex, bufWriteLinked, normalize):
    getstickspectrum2fromh5.argtypes = [ct.c_char_p, _float_array, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int8, ct.c_int8]
    return getstickspectrum2fromh5(filename, spectrum, segmentIndex, segmentEndIndex, bufIndex, bufEndIndex, writeIndex, writeEndIndex, bufWriteLinked, normalize)

getpeakparametersfromh5 = h5lib.TwGetPeakParametersFromH5 if os.name=='posix' else h5lib._TwGetPeakParametersFromH5
def TwGetPeakParametersFromH5(filename, peakPar, peakIndex):
    getpeakparametersfromh5.argtypes = [ct.c_char_p, ct.POINTER(TPeakPar), ct.c_int]
    # return getpeakparametersfromh5(filename, ct.pointer(peakPar), peakIndex)
    return getpeakparametersfromh5(filename, peakPar, peakIndex)

getspecxaxisfromh5 = h5lib.TwGetSpecXaxisFromH5 if os.name=='posix' else h5lib._TwGetSpecXaxisFromH5
def TwGetSpecXaxisFromH5(filename, specAxis, axisType, unitLabel, maxMass, writeIndex):
    getspecxaxisfromh5.argtypes = [ct.c_char_p, _double_array, ct.c_int, ct.c_char_p, ct.c_double, ct.c_int]
    return getspecxaxisfromh5(filename, specAxis, axisType, unitLabel, maxMass, writeIndex)

getsegmentprofilefromh5 = h5lib.TwGetSegmentProfileFromH5 if os.name=='posix' else h5lib._TwGetSegmentProfileFromH5
def TwGetSegmentProfileFromH5(filename, segmentProfile, peakIndex, bufStartIndex, bufEndIndex, writeStartIndex, writeEndIndex, bufWriteLinked):
    getsegmentprofilefromh5.argtypes = [ct.c_char_p, _float_array, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int8]
    return getsegmentprofilefromh5(filename, segmentProfile, peakIndex, bufStartIndex, bufEndIndex, writeStartIndex, writeEndIndex, bufWriteLinked)

getsegmentprofile2fromh5 = h5lib.TwGetSegmentProfile2FromH5 if os.name=='posix' else h5lib._TwGetSegmentProfile2FromH5
def TwGetSegmentProfile2FromH5(filename, segmentProfile, peakIndex, bufStartIndex, bufEndIndex, writeStartIndex, writeEndIndex, bufWriteLinked):
    getsegmentprofile2fromh5.argtypes = [ct.c_char_p, _float_array, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int8]
    return getsegmentprofile2fromh5(filename, segmentProfile, peakIndex, bufStartIndex, bufEndIndex, writeStartIndex, writeEndIndex, bufWriteLinked)

getbufwriteprofilefromh5 = h5lib.TwGetBufWriteProfileFromH5 if os.name=='posix' else h5lib._TwGetBufWriteProfileFromH5
def TwGetBufWriteProfileFromH5(filename, profile, peakIndex, segmentStartIndex, segmentEndIndex):
    getbufwriteprofilefromh5.argtypes = [ct.c_char_p, _float_array, ct.c_int, ct.c_int, ct.c_int]
    return getbufwriteprofilefromh5(filename, profile, peakIndex, segmentStartIndex, segmentEndIndex)

getbufwriteprofile2fromh5 = h5lib.TwGetBufWriteProfile2FromH5 if os.name=='posix' else h5lib._TwGetBufWriteProfile2FromH5
def TwGetBufWriteProfile2FromH5(filename, profile, peakIndex, segmentStartIndex, segmentEndIndex):
    getbufwriteprofile2fromh5.argtypes = [ct.c_char_p, _float_array, ct.c_int, ct.c_int, ct.c_int]
    return getbufwriteprofile2fromh5(filename, profile, peakIndex, segmentStartIndex, segmentEndIndex)

getreguserdatafromh5 = h5lib.TwGetRegUserDataFromH5 if os.name=='posix' else h5lib._TwGetRegUserDataFromH5
def TwGetRegUserDataFromH5(filename, location, bufIndex, writeIndex, bufLength, dataBuffer, description):
    if dataBuffer is None:
        getreguserdatafromh5.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_int, ndpointer(np.int32), ct.c_void_p, ct.c_char_p]
    else:
        getreguserdatafromh5.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_int, ndpointer(np.int32), _double_array, ct.c_char_p]
    return getreguserdatafromh5(filename, location, bufIndex, writeIndex, bufLength, dataBuffer, description)

gettofdata = h5lib.TwGetTofData if os.name=='posix' else h5lib._TwGetTofData
def TwGetTofData(filename, sampleOffset, sampleCount, segOffset, segCount, bufOffset,  bufCount,  writeOffset,  writeCount, dataBuffer):
    gettofdata.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, _float_array]
    return gettofdata(filename, sampleOffset, sampleCount, segOffset, segCount, bufOffset,  bufCount,  writeOffset,  writeCount, dataBuffer)

gettofdata2 = h5lib.TwGetTofData2 if os.name=='posix' else h5lib._TwGetTofData2
def TwGetTofData2(filename, sampleOffset, sampleCount, segOffset, segCount, bufOffset,  bufCount,  writeOffset,  writeCount, dataBuffer):
    gettofdata2.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, _float_array]
    return gettofdata2(filename, sampleOffset, sampleCount, segOffset, segCount, bufOffset,  bufCount,  writeOffset,  writeCount, dataBuffer)

gettimingdata = h5lib.TwGetTimingData if os.name=='posix' else h5lib._TwGetTimingData
def TwGetTimingData(filename, bufOffset, bufCount, writeOffset, writeCount, dataBuffer):
    gettimingdata.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, _double_array]
    return gettimingdata(filename, bufOffset, bufCount, writeOffset, writeCount, dataBuffer)

getpeakdata = h5lib.TwGetPeakData if os.name=='posix' else h5lib._TwGetPeakData
def TwGetPeakData(filename, peakOffset, peakCount, segOffset, segCount, bufOffset,  bufCount,  writeOffset,  writeCount, dataBuffer):
    getpeakdata.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, _float_array]
    return getpeakdata(filename, peakOffset, peakCount, segOffset, segCount, bufOffset,  bufCount,  writeOffset,  writeCount, dataBuffer)

getpeakdata2 = h5lib.TwGetPeakData2 if os.name=='posix' else h5lib._TwGetPeakData2
def TwGetPeakData2(filename, peakOffset, peakCount, segOffset, segCount, bufOffset,  bufCount,  writeOffset,  writeCount, dataBuffer):
    getpeakdata2.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, _float_array]
    return getpeakdata2(filename, peakOffset, peakCount, segOffset, segCount, bufOffset,  bufCount,  writeOffset,  writeCount, dataBuffer)

getintattributefromh5 = h5lib.TwGetIntAttributeFromH5 if os.name=='posix' else h5lib._TwGetIntAttributeFromH5
def TwGetIntAttributeFromH5(filename, location, name, attribute):
    getintattributefromh5.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p, ndpointer(dtype=np.int32, flags='CONTIGUOUS', shape=1)]
    return getintattributefromh5(filename, location, name, attribute)

getuintattributefromh5 = h5lib.TwGetUintAttributeFromH5 if os.name=='posix' else h5lib._TwGetUintAttributeFromH5
def TwGetUintAttributeFromH5(filename, location, name, attribute):
    getuintattributefromh5.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p, ndpointer(dtype=np.uint32, flags='CONTIGUOUS', shape=1)]
    return getuintattributefromh5(filename, location, name, attribute)

getint64attributefromh5 = h5lib.TwGetInt64AttributeFromH5 if os.name=='posix' else h5lib._TwGetInt64AttributeFromH5
def TwGetInt64AttributeFromH5(filename, location, name, attribute):
    getint64attributefromh5.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p, ndpointer(dtype=np.int64, flags='CONTIGUOUS', shape=1)]
    return getint64attributefromh5(filename, location, name, attribute)

getuint64attributefromh5 = h5lib.TwGetUint64AttributeFromH5 if os.name=='posix' else h5lib._TwGetUint64AttributeFromH5
def TwGetUint64AttributeFromH5(filename, location, name, attribute):
    getuint64attributefromh5.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p, ndpointer(dtype=np.uint64, flags='CONTIGUOUS', shape=1)]
    return getuint64attributefromh5(filename, location, name, attribute)

getfloatattributefromh5 = h5lib.TwGetFloatAttributeFromH5 if os.name=='posix' else h5lib._TwGetFloatAttributeFromH5
def TwGetFloatAttributeFromH5(filename, location, name, attribute):
    getfloatattributefromh5.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p, ndpointer(dtype=np.float32, flags='CONTIGUOUS', shape=1)]
    return getfloatattributefromh5(filename, location, name, attribute)

getdoubleattributefromh5 = h5lib.TwGetDoubleAttributeFromH5 if os.name=='posix' else h5lib._TwGetDoubleAttributeFromH5
def TwGetDoubleAttributeFromH5(filename, location, name, attribute):
    getdoubleattributefromh5.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p, ndpointer(dtype=np.double, flags='CONTIGUOUS', shape=1)]
    return getdoubleattributefromh5(filename, location, name, attribute)

getstringattributefromh5 = h5lib.TwGetStringAttributeFromH5 if os.name=='posix' else h5lib._TwGetStringAttributeFromH5
def TwGetStringAttributeFromH5(filename, location, name, attribute):
    getstringattributefromh5.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_char_p]
    return getstringattributefromh5(filename, location, name, attribute)

setintattributeinh5 = h5lib.TwSetIntAttributeInH5 if os.name=='posix' else h5lib._TwSetIntAttributeInH5
def TwSetIntAttributeInH5(filename, location, name, attribute):
    setintattributeinh5.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_int]
    return setintattributeinh5(filename, location, name, attribute)

setuintattributeinh5 = h5lib.TwSetUintAttributeInH5 if os.name=='posix' else h5lib._TwSetUintAttributeInH5
def TwSetUintAttributeInH5(filename, location, name, attribute):
    setuintattributeinh5.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_uint]
    return setuintattributeinh5(filename, location, name, attribute)

setint64attributeinh5 = h5lib.TwSetInt64AttributeInH5 if os.name=='posix' else h5lib._TwSetInt64AttributeInH5
def TwSetInt64AttributeInH5(filename, location, name, attribute):
    setint64attributeinh5.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_int64]
    return setint64attributeinh5(filename, location, name, attribute)

setuint64attributeinh5 = h5lib.TwSetUint64AttributeInH5 if os.name=='posix' else h5lib._TwSetUint64AttributeInH5
def TwSetUint64AttributeInH5(filename, location, name, attribute):
    setuint64attributeinh5.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_uint64]
    return setuint64attributeinh5(filename, location, name, attribute)

setfloatattributeinh5 = h5lib.TwSetFloatAttributeInH5 if os.name=='posix' else h5lib._TwSetFloatAttributeInH5
def TwSetFloatAttributeInH5(filename, location, name, attribute):
    setfloatattributeinh5.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_float]
    return setfloatattributeinh5(filename, location, name, attribute)

setdoubleattributeinh5 = h5lib.TwSetDoubleAttributeInH5 if os.name=='posix' else h5lib._TwSetDoubleAttributeInH5
def TwSetDoubleAttributeInH5(filename, location, name, attribute):
    setdoubleattributeinh5.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_double]
    return setdoubleattributeinh5(filename, location, name, attribute)

setstringattributeinh5 = h5lib.TwSetStringAttributeInH5 if os.name=='posix' else h5lib._TwSetStringAttributeInH5
def TwSetStringAttributeInH5(filename, location, name, attribute):
    setstringattributeinh5.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_char_p]
    return setstringattributeinh5(filename, location, name, attribute)

deleteattributeinh5 = h5lib.TwDeleteAttributeInH5 if os.name=='posix' else h5lib._TwDeleteAttributeInH5
def TwDeleteAttributeInH5(filename, location, name):
    deleteattributeinh5.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p]
    return deleteattributeinh5(filename, location, name)    


    
changepeakdatainit = h5lib.TwChangePeakDataInit if os.name=='posix' else h5lib._TwChangePeakDataInit
def TwChangePeakDataInit(filename, newPeakPar, nbrNewPeakPar, options):
    changepeakdatainit.argtypes = [ct.c_char_p, ct.POINTER(TPeakPar), ct.c_int, ct.c_int]
    return changepeakdatainit(filename, newPeakPar, nbrNewPeakPar, options)

changepeakdatawrite = h5lib.TwChangePeakDataWrite if os.name=='posix' else h5lib._TwChangePeakDataWrite
def TwChangePeakDataWrite(filename, peakOffset, peakCount, segOffset, segCount, bufOffset, bufCount, writeOffset, writeCount, data, data2):
    changepeakdatawrite.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
                                    ct.c_void_p if data is None else ndpointer(dtype=np.float32, shape=segCount*bufCount*writeCount),
                                    ct.c_void_p if data2 is None else ndpointer(dtype=np.float32, shape=segCount*bufCount*writeCount)]
    return changepeakdatawrite(filename, peakOffset, peakCount, segOffset, segCount, bufOffset, bufCount, writeOffset, writeCount, data, data2)

changepeakdatafinalize = h5lib.TwChangePeakDataFinalize if os.name=='posix' else h5lib._TwChangePeakDataFinalize
def TwChangePeakDataFinalize(filename):
    changepeakdatafinalize.argtypes = [ct.c_char_p]
    return changepeakdatafinalize(filename)
    
    

TwProgressCallback = ct.CFUNCTYPE(ct.c_bool, ct.c_double)

changepeaktable = h5lib.TwChangePeakTable if os.name=='posix' else h5lib._TwChangePeakTable
def TwChangePeakTable(filename, newPeakPar, nbrNewPeakPar, compressionLevel, callback):
    if callback is None:
        changepeaktable.argtypes = [ct.c_char_p, ct.POINTER(TPeakPar), ct.c_int, ct.c_int, ct.c_void_p]
    else:
        changepeaktable.argtypes = [ct.c_char_p, ct.POINTER(TPeakPar), ct.c_int, ct.c_int, TwProgressCallback]
    return changepeaktable(filename, newPeakPar, nbrNewPeakPar, compressionLevel, callback)

changepeaktablefromfile = h5lib.TwChangePeakTableFromFile if os.name=='posix' else h5lib._TwChangePeakTableFromFile
def TwChangePeakTableFromFile (filename, massTable, compressionLevel, callback):
    if callback is None:
        changepeaktablefromfile.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_void_p]
    else:
        changepeaktablefromfile.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_int, TwProgressCallback]
    return changepeaktablefromfile(filename, massTable, compressionLevel, callback)

getbuftimefromh5 = h5lib.TwGetBufTimeFromH5 if os.name=='posix' else h5lib._TwGetBufTimeFromH5
def TwGetBufTimeFromH5(filename, bufTime, bufIndex, writeIndex):
    getbuftimefromh5.argtypes = [ct.c_char_p, ndpointer(dtype=np.double, shape=1), ct.c_int, ct.c_int]
    return getbuftimefromh5(filename, bufTime, bufIndex, writeIndex)

h5setmasscalib = h5lib.TwH5SetMassCalib if os.name=='posix' else h5lib._TwH5SetMassCalib
def TwH5SetMassCalib(filename, mode, nbrParams, p, nbrPoints, mass, tof, weight):
    h5setmasscalib.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, _double_array, ct.c_int, _double_array, _double_array, _double_array]
    return h5setmasscalib(filename, mode, nbrParams, p, nbrPoints, mass, tof, weight)

h5setmasscalib2 = h5lib.TwH5SetMassCalib2 if os.name=='posix' else h5lib._TwH5SetMassCalib2
def TwH5SetMassCalib2(filename, mode, nbrParams, p, nbrPoints, mass, tof, weight):
    h5setmasscalib2.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, _double_array, ct.c_int, _double_array, _double_array, _double_array]
    return h5setmasscalib2(filename, mode, nbrParams, p, nbrPoints, mass, tof, weight)

h5setmasscalibex = h5lib.TwH5SetMassCalibEx if os.name=='posix' else h5lib._TwH5SetMassCalibEx
def TwH5SetMassCalibEx(filename, mode, nbrParams, p, nbrPoints, mass, tof, weight, label):
    h5setmasscalibex.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, _double_array, ct.c_int, _double_array, _double_array, _double_array, ct.c_char_p]
    return h5setmasscalibex(filename, mode, nbrParams, p, nbrPoints, mass, tof, weight, label)

h5setmasscalib2ex = h5lib.TwH5SetMassCalib2Ex if os.name=='posix' else h5lib._TwH5SetMassCalib2Ex
def TwH5SetMassCalib2Ex(filename, mode, nbrParams, p, nbrPoints, mass, tof, weight, label):
    h5setmasscalib2ex.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, _double_array, ct.c_int, _double_array, _double_array, _double_array, ct.c_char_p]
    return h5setmasscalib2ex(filename, mode, nbrParams, p, nbrPoints, mass, tof, weight, label)

h5getmasscalibpar = h5lib.TwH5GetMassCalibPar if os.name=='posix' else h5lib._TwH5GetMassCalibPar
def TwH5GetMassCalibPar(filename, segmentIndex, bufIndex, writeIndex, mode, nbrParams, p):
    h5getmasscalibpar.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, ct.c_int, ndpointer(dtype=ct.c_int32, shape=1), ndpointer(dtype=ct.c_int32, shape=1), ndpointer(dtype=np.float64)]
    return h5getmasscalibpar(filename, segmentIndex, bufIndex, writeIndex, mode, nbrParams, p)        
	
getacquisitionlogfromh5 = h5lib.TwGetAcquisitionLogFromH5 if os.name=='posix' else h5lib._TwGetAcquisitionLogFromH5
def TwGetAcquisitionLogFromH5(location, index, timestamp, logText):
    getacquisitionlogfromh5.argtypes = [ct.c_char_p, ct.c_int, ndpointer(dtype=np.int64), ct.c_char_p]
    return getacquisitionlogfromh5(location, index, timestamp, logText)

geteventlistdatafromh5 = h5lib.TwGetEventListDataFromH5 if os.name=='posix' else h5lib._TwGetEventListDataFromH5
def TwGetEventListDataFromH5(filename, segStartIndex, segEndIndex, bufStartIndex, bufEndIndex, writeStartIndex, writeEndIndex):
    uintptr = ct.POINTER(ct.c_uint)
    dataLength = uintptr()
    dataBuffer = ct.POINTER(uintptr)()
    rv = geteventlistdatafromh5(filename, segStartIndex, segEndIndex, bufStartIndex, bufEndIndex, writeStartIndex, writeEndIndex, ct.byref(dataBuffer), ct.byref(dataLength))
    allTimestamps = []
    allSamples = []
    if rv == 4:
        nbrSpectra = (segEndIndex-segStartIndex+1)*(bufEndIndex-bufStartIndex+1)*(writeEndIndex-writeStartIndex+1)
        data = [np.ctypeslib.as_array(dataBuffer[i], shape=(dataLength[i],)) if dataLength[i] > 0 else np.empty((0,), dtype=np.uint32) for i in range(nbrSpectra)]
        for i in range(nbrSpectra):
            timeStamps = []
            samples = []
            nEl = dataLength[i]
            index = 0
            while index < nEl:
                ts = data[i][index]&0xFFFFFF
                timeStamps.append(ts)
                nSamples = data[i][index]>>24
                index += 1
                if nSamples > 0:
                    samples.append(np.frombuffer(data[i][index:index+nSamples], dtype=np.float32))
                    index += nSamples
                else:
                    samples.append(np.empty((0,), dtype=np.float32) )
            allTimestamps.append(timeStamps)
            allSamples.append(samples)
    return (allTimestamps, allSamples)

freeeventlistdata = h5lib.TwFreeEventListData if os.name=='posix' else h5lib._TwFreeEventListData
def TwFreeEventListData():
    return freeeventlistdata()

addlogentry = h5lib.TwH5AddLogEntry if os.name=='posix' else h5lib._TwH5AddLogEntry
def TwH5AddLogEntry(filename, logEntryText, logEntryTime):
    addlogentry.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_uint64]
    return addlogentry(filename, logEntryText, logEntryTime)
    
getuserdatafromh5 = h5lib.TwGetUserDataFromH5 if os.name == 'posix' else h5lib._TwGetUserDataFromH5
def TwGetUserDataFromH5(filename, location, rowIndex, nbrElements, buf, elementDescription=None):
    def bytesToStrings(bytes):
        """ Takes a numpy array with data type uint8 with "TofDaq" format and converts it to a list of python strings
            The "TofDaq" format is a byte array where each string is 256 bytes long (or ended by a null)
        """
        strings = []
        for i in range(0, len(bytes), 256):
            s = bytes[i:i+256].tostring()
            strings.append(s.split('\0')[0])
        return strings
    if buf is None:
        getuserdatafromh5.argtypes = [ct.c_char_p
                                      ,ct.c_char_p
                                      ,ndpointer(dtype=ct.c_int32, shape=1)
                                      ,ndpointer(dtype=ct.c_int32, shape=1)
                                      ,ct.c_void_p
                                      ,ct.c_void_p]
    elif elementDescription is None:
        getuserdatafromh5.argtypes = [ct.c_char_p
                                      ,ct.c_char_p
                                      ,ndpointer(dtype=ct.c_int32, shape=1)
                                      ,ndpointer(dtype=ct.c_int32, shape=1)
                                      ,_double_array
                                      ,ct.c_void_p]
    else:
        getuserdatafromh5.argtypes = [ct.c_char_p
                                      ,ct.c_char_p
                                      ,ndpointer(dtype=ct.c_int32, shape=1)
                                      ,ndpointer(dtype=ct.c_int32, shape=1)
                                      ,_double_array
                                      ,ndpointer(dtype=ct.c_uint8)]
        if type(elementDescription) == list:
            tempdesc = np.zeros(256 * nbrElements[0], dtype=np.uint8)
            ret = getuserdatafromh5(filename, location, rowIndex, nbrElements, buf, tempdesc)
            elementDescription[:] = bytesToStrings(tempdesc)
            return ret
    return getuserdatafromh5(filename, location, rowIndex, nbrElements, buf, elementDescription)


readrawdata = h5lib.TwReadRawData if os.name == 'posix' else h5lib._TwReadRawData
def TwReadRawData(filename, channel, bufIndex, writeIndex, bufferSize, buffer):
    if buffer is None:
        readrawdata.argtypes = [ct.c_char_p, ct.c_int32, ct.c_int32, ct.c_int32, ndpointer(dtype=ct.c_int32, shape=1), ct.c_void_p] 
    else:
        readrawdata.argtypes = [ct.c_char_p, ct.c_int32, ct.c_int32, ct.c_int32, ndpointer(dtype=ct.c_int32, shape=1), ndpointer()]
    return readrawdata(filename, channel, bufIndex, writeIndex, bufferSize, buffer)
    

#TOFWERK_H5_API TwRetVal TwGenerateSegmentProfilesFromEventList(char* Filename, int nbrProfiles, double* startMass, double* endMass, int bufStartIndex, int bufEndIndex, int writeStartIndex, int writeEndIndex, float* data, bool startEndInSamples);    
generatesegmentprofilesfromeventlist = h5lib.TwGenerateSegmentProfilesFromEventList if os.name == 'posix' else h5lib._TwGenerateSegmentProfilesFromEventList
def TwGenerateSegmentProfilesFromEventList(filename, nbrProfiles, startMass, endMass, bufStartIndex, bufEndIndex, writeStartIndex, writeEndIndex, data, startEndInSamples):
    generatesegmentprofilesfromeventlist.argtypes = [ct.c_char_p, ct.c_int32, _double_array, _double_array, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, _float_array,  ct.c_int8]
    return generatesegmentprofilesfromeventlist(filename, nbrProfiles, startMass, endMass, bufStartIndex, bufEndIndex, writeStartIndex, writeEndIndex, data, startEndInSamples)

#TOFWERK_H5_API TwRetVal TwGetRegUserDataSourcesFromH5(char* filename, int* nbrSources, char* sourceLocation, int* sourceLength, bool* hasDesc, int* type) {
getreguserdatasources = h5lib.TwGetRegUserDataSourcesFromH5 if os.name == 'posix' else h5lib._TwGetRegUserDataSourcesFromH5
def TwGetRegUserDataSourcesFromH5Py(filename):
    getreguserdatasources.argtypes = [ct.c_char_p, ndpointer(dtype=ct.c_int32, shape=1), ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p]
    nbrSources = np.zeros((1,), dtype=ct.c_int32)
    rv = getreguserdatasources(filename, nbrSources, None, None, None, None)
    if rv != TwValueAdjusted:
        return []
    getreguserdatasources.argtypes = [ct.c_char_p, ndpointer(dtype=ct.c_int32, shape=1), ct.c_char_p, ndpointer(dtype=ct.c_int32), ndpointer(dtype=ct.c_uint8), ndpointer(dtype=ct.c_int32)]
    srcLocation = ct.create_string_buffer(int(nbrSources[0]*256))
    srcLength = np.ndarray(nbrSources, dtype=ct.c_int32)
    srcHasDesc = np.ndarray(nbrSources, dtype=ct.c_uint8)
    srcType = np.ndarray(nbrSources, dtype=ct.c_int32)
    rv = getreguserdatasources(filename, nbrSources, srcLocation, srcLength, srcHasDesc, srcType)
    result = []
    if rv != TwSuccess:
        return result
    for i in range(nbrSources[0]):
        result.append((srcLocation[i*256:(i+1)*256].strip(b'\0'),
                       srcLength[i],
                       False if srcHasDesc[i] == 0 else True,
                        srcType[i]))
    return result

    
    
#TOFWERK_H5_API TwRetVal TwH5MakePaletteImage(char* filename, char* location, float* data, int width, int height, int palette, unsigned char paletteOffset, bool paletteInvert, float* dataMinMax, float gammaVal);
makepaletteimage = h5lib.TwH5MakePaletteImage if os.name=='posix' else h5lib._TwH5MakePaletteImage
def TwH5MakePaletteImage(filename, location, data, palette, paletteOffset, paletteInvert, dataMinMax, gammaVal):
    if data.ndim != 2:
        raise ValueError('Data array must be 2D!')
    height, width = data.shape
    makepaletteimage.argtypes = [ct.c_char_p, ct.c_char_p, ndpointer(dtype=np.float32, shape=(height*width,)), ct.c_int32, ct.c_int32, ct.c_int32, ct.c_uint8, ct.c_int8,
                                 ct.c_void_p if dataMinMax is None else ndpointer(dtype=np.float32, shape=(2)), ct.c_float]
    return makepaletteimage(filename, location, data.flatten(), width, height, palette, paletteOffset, paletteInvert, dataMinMax, gammaVal)



#TOFWERK_H5_API TwRetVal TwGenerateBufWriteProfilesFromEventList(char* Filename, int nbrProfiles, double* startMass, double* endMass, int segStartIndex, int segEndIndex, float* data, bool startEndInSamples);
generatebufwriteprofilesfromeventlist = h5lib.TwGenerateBufWriteProfilesFromEventList if os.name == 'posix' else h5lib._TwGenerateBufWriteProfilesFromEventList
def TwGenerateBufWriteProfilesFromEventList(filename, nbrProfiles, startMass, endMass, segStartIndex, segEndIndex, data, startEndInSamples):
    generatebufwriteprofilesfromeventlist.argtypes = [ct.c_char_p, ct.c_int32, _double_array, _double_array, ct.c_int32, ct.c_int32, _float_array,  ct.c_bool]
    return generatebufwriteprofilesfromeventlist(filename, nbrProfiles, startMass, endMass, segStartIndex, segEndIndex, data, startEndInSamples)


