import ctypes as ct
from numpy.ctypeslib import ndpointer
import numpy as np


_double_array = ndpointer(dtype=np.float64, flags='CONTIGUOUS')
_float_array = ndpointer(dtype=np.float32, flags='CONTIGUOUS')

libTofDaq = ct.CDLL('./TofDaqDll.dll')

class TSharedMemoryDesc(ct.Structure):
    _fields_ = [("nbrSamples", ct.c_int),
                ("nbrRawSamples", ct.c_int),
                ("nbrPeaks", ct.c_int),
                ("nbrWaveforms", ct.c_int),
                ("nbrSegments", ct.c_int),
                ("nbrBlocks", ct.c_int),
                ("nbrMemories", ct.c_int),
                ("nbrBufs", ct.c_int),
                ("nbrWrites", ct.c_int),
                ("nbrRuns", ct.c_int),
                ("iWaveform", ct.c_int),
                ("iSegment", ct.c_int),
                ("iBlock", ct.c_int),
                ("iMemory", ct.c_int),
                ("iBuf", ct.c_int),
                ("iWrite", ct.c_int),
                ("iRun", ct.c_int),
                ("totalBufsRecorded", ct.c_int),
                ("totalBufsProcessed", ct.c_int),
                ("totalBufsWritten", ct.c_int),
                ("overallBufsProcessed", ct.c_int),
                ("totalNbrMemories",  ct.c_int),
                ("totalMemoriesProcessed",  ct.c_int),
                ("rawDataRecordedBuf1", ct.c_uint),
                ("rawDataRecordedBuf2", ct.c_uint),
                ("rawDataLastElementInBuffer1", ct.c_uint),
                ("rawDataLastElementInBuffer2", ct.c_uint),
                ("rawDataProcessedBuf1", ct.c_uint),
                ("rawDataProcessedBuf2", ct.c_uint),
                ("rawDataWrittenBuf1", ct.c_uint),
                ("rawDataWrittenBuf2", ct.c_uint),
                ("sampleInterval", ct.c_float),
                ("tofPeriod", ct.c_int),
                ("nbrCubes", ct.c_int),
                ("blockPeriod", ct.c_longlong),
                ("blockPulseDelay", ct.c_longlong),
                ("blockDelay", ct.c_longlong),
                ("singleIonSignal", ct.c_float),
                ("singleIonSignal2", ct.c_float),
                ("massCalibMode", ct.c_int),
                ("massCalibMode2", ct.c_int),
                ("nbrMassCalibParams", ct.c_int),
                ("nbrMassCalibParams2", ct.c_int),
                ("p", ct.c_double*16),
                ("p2", ct.c_double*16),
                ("R0", ct.c_float),
                ("dm", ct.c_float),
                ("m0", ct.c_float),
                ("secondTof", ct.c_bool),
                ("chIniFileName", ct.c_char*256),
                ("currentDataFileName", ct.c_char*256),
                ("segIlf", ct.c_ubyte),
                ("iCube", ct.c_ushort),
                ("daqMode", ct.c_int),
                ("acquisitionMode", ct.c_int),
                ("combineMode", ct.c_int),
                ("recalibFreq", ct.c_int),
                ("acquisitionLogText", ct.c_char*256),
                ("acquisitionLogTime", ct.c_ulonglong),
                ("timeZero", ct.c_ulonglong),
                ("externalLock", ct.c_void_p),
                ("processingLevel", ct.c_uint),
                ("attributeType", ct.c_int),
                ("attributeObject", ct.c_char*256),
                ("attributeName", ct.c_char*128),
                ("attributeInt", ct.c_int),
                ("attributeDouble", ct.c_double),
                ("enableVarNbrMemories", ct.c_int),
                ("nbrSteps", ct.c_int),
                ("currentStepAtBuf", ct.c_int),
                ("nbrMemoriesForCurrentStep", ct.c_int)]

class TPeakPar(ct.Structure):
    _fields_ = [("label", ct.c_char*64),
                ("mass", ct.c_float),
                ("loMass", ct.c_float),
                ("hiMass", ct.c_float)]

class TSharedMemoryPointer(ct.Structure):
    _fields_ = [("sumSpectrum", ct.POINTER(ct.c_double)),
                ("sumSpectrum2", ct.POINTER(ct.c_double)),
                ("tofData", ct.POINTER(ct.POINTER(ct.c_float))),
                ("tofData2", ct.POINTER(ct.POINTER(ct.c_float))),
                ("peakData", ct.POINTER(ct.c_float)),
                ("peakData2", ct.POINTER(ct.c_float)),
                ("timing", ct.POINTER(ct.c_double)),
                ("rawData32Ch1", ct.POINTER(ct.c_uint32)),
                ("rawData16Ch1", ct.POINTER(ct.c_uint16)),
                ("rawData8Ch1", ct.POINTER(ct.c_int8)), 
                ("rawData32Ch2", ct.POINTER(ct.c_uint32)),
                ("rawData16Ch2", ct.POINTER(ct.c_uint16)),
                ("rawData8Ch2", ct.POINTER(ct.c_int8))]


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


#--------------------- CONTROL FUNCTIONS ---------------------------------------
def TwGetDllVersion():
    libTofDaq._TwGetDllVersion.restype = ct.c_double 
    return libTofDaq._TwGetDllVersion()

TOFDAQDLL_REV = int(1.0E6*(TwGetDllVersion() - 1.99) + 0.5)

def TwInitializeDll():
    return libTofDaq._TwInitializeDll()

def TwCleanupDll():
    libTofDaq._TwCleanupDll()

def TwTofDaqRunning():
    libTofDaq._TwTofDaqRunning.restype = ct.c_bool
    return libTofDaq._TwTofDaqRunning()

def TwDaqActive():
    libTofDaq._TwDaqActive.restype = ct.c_bool
    return libTofDaq._TwDaqActive()

def TwStartAcquisition():
    return libTofDaq._TwStartAcquisition()

def TwStopAcquisition():
    return libTofDaq._TwStopAcquisition()

def TwContinueAcquisition():
    return libTofDaq._TwContinueAcquisition()

def TwManualContinueNeeded():
    if libTofDaq._TwManualContinueNeeded(): return True
    else: return False

def TwCloseTofDaqRec():
    return libTofDaq._TwCloseTofDaqRec() 

def TwLockBuf(timeout, bufToLock):
    libTofDaq._TwLockBuf.argtypes = [ct.c_int, ct.c_int]
    return libTofDaq._TwLockBuf(timeout, bufToLock)

def TwUnLockBuf(bufToUnlock):
    libTofDaq._TwUnLockBuf.argtypes = [ct.c_int]
    return libTofDaq._TwUnLockBuf(bufToUnlock)
	
def TwIssueDio4Pulse(delay, width):
    libTofDaq._TwIssueDio4Pulse.argtypes = [ct.c_int, ct.c_int]
    return libTofDaq._TwIssueDio4Pulse(delay, width)

def TwSetDio4State(state):
    libTofDaq._TwSetDio4State.argtypes = [ct.c_int]
    return libTofDaq._TwSetDio4State(state)

if TOFDAQDLL_REV >= 1204:
    def TwWaitingForDioStartSignal():
        libTofDaq._TwWaitingForDioStartSignal.restype = ct.c_bool
        return libTofDaq._TwWaitingForDioStartSignal()

    def TwSendDioStartSignal():
        return libTofDaq._TwSendDioStartSignal()

#----------------------- CONFIGURATION FUNCTIONS -------------------------------

def TwShowConfigWindow(configWindowIndex):
    return libTofDaq._TwShowConfigWindow(configWindowIndex)

def TwLoadIniFile(iniFilename):
    libTofDaq._TwLoadIniFile.argtypes = [ct.c_char_p]
    return libTofDaq._TwLoadIniFile(iniFilename)

def TwSaveIniFile(iniFilename):
    libTofDaq._TwLoadIniFile.argtypes = [ct.c_char_p]
    return libTofDaq._TwSaveIniFile(iniFilename)

def TwGetDaqParameter(Parameter):
    libTofDaq._TwGetDaqParameter.argtypes = [ct.c_char_p]
    libTofDaq._TwGetDaqParameter.restype = ct.c_char_p
    return libTofDaq._TwGetDaqParameter(Parameter)

def TwGetDaqParameterInt(Parameter):
    libTofDaq._TwGetDaqParameterInt.argtypes = [ct.c_char_p]
    libTofDaq._TwGetDaqParameterInt.restype = ct.c_int
    return int(libTofDaq._TwGetDaqParameterInt(Parameter))

def TwGetDaqParameterBool(Parameter):
    libTofDaq._TwGetDaqParameterBool.argtypes = [ct.c_char_p]
    libTofDaq._TwGetDaqParameterBool.restype = ct.c_bool
    return libTofDaq._TwGetDaqParameterBool(Parameter)

def TwGetDaqParameterFloat(Parameter):
    libTofDaq._TwGetDaqParameterFloat.argtypes = [ct.c_char_p]
    libTofDaq._TwGetDaqParameterFloat.restype = ct.c_float
    return float(libTofDaq._TwGetDaqParameterFloat(Parameter))

def TwGetDaqParameterInt64(Parameter):
    libTofDaq._TwGetDaqParameterInt64.argtypes = [ct.c_char_p]
    libTofDaq._TwGetDaqParameterInt64.restype = ct.c_int64
    return int(libTofDaq._TwGetDaqParameterInt64(Parameter))

def TwGetDaqParameterDouble(Parameter):
    libTofDaq._TwGetDaqParameterFloat.argtypes = [ct.c_char_p]
    libTofDaq._TwGetDaqParameterFloat.restype = ct.c_double
    return float(libTofDaq._TwGetDaqParameterDouble(Parameter))

def TwGetDaqParameterIntRef(Parameter, pValue):
    libTofDaq._TwGetDaqParameterInt.argtypes = [ct.c_char_p, ndpointer(np.int32, shape=1)]
    return libTofDaq._TwGetDaqParameterIntRef(Parameter, pValue)

# Here a byte has to be used since there are no numpy bool arrays
def TwGetDaqParameterBoolRef(Parameter, pValue):
    libTofDaq._TwGetDaqParameterBoolRef.argtypes = [ct.c_char_p, ndpointer(c_bool, shape=1)]
    return libTofDaq._TwGetDaqParameterBoolRef(Parameter, pValue)

def TwGetDaqParameterFloatRef(Parameter, pValue):
    libTofDaq._TwGetDaqParameterFloatRef.argtypes = [ct.c_char_p, ndpointer(np.float32, shape=1)]
    return libTofDaq._TwGetDaqParameterFloatRef(Parameter, pValue)

def TwGetDaqParameterInt64Ref(Parameter, pValue):
    libTofDaq._TwGetDaqParameterInt64.argtypes = [ct.c_char_p, ndpointer(np.int64, shape=1)]
    return libTofDaq._TwGetDaqParameterInt64Ref(Parameter, pValue)

def TwGetDaqParameterDoubleRef(Parameter, pValue):
    libTofDaq._TwGetDaqParameterDoubleRef.argtypes = [ct.c_char_p, ndpointer(np.float64, shape=1)]
    return libTofDaq._TwGetDaqParameterDoubleRef(Parameter, pValue)

# This function is a bit complicated to use. It needs a numpy array of size 256 np.uint8.
# One then have to parse this manually looking for the null.
# Consult numpy.ndarray.tostring function for a hint to get started
def TwGetDaqParameterStringRef(Parameter, pValue):
    if isinstance(pValue, c_char_p):
        libTofDaq._TwGetDaqParameterStringRef.argtypes = [ct.c_char_p, ct.c_char_p]
        return libTofDaq._TwGetDaqParameterStringRef(Parameter, pValue)
    else:
        libTofDaq._TwGetDaqParameterStringRef.argtypes = [ct.c_char_p, ndpointer(np.uint8, shape=256) ]
        return libTofDaq._TwGetDaqParameterStringRef(Parameter, pValue)



def TwSetDaqParameter(Parameter, Value):
    libTofDaq._TwSetDaqParameter.argtypes = [ct.c_char_p, ct.c_char_p]
    return libTofDaq._TwSetDaqParameter(Parameter, Value)

def TwSetDaqParameterInt(Parameter, Value):
    libTofDaq._TwSetDaqParameterInt.argtypes = [ct.c_char_p, ct.c_int]
    return libTofDaq._TwSetDaqParameterInt(Parameter, Value)

def TwSetDaqParameterBool(Parameter, Value):
    libTofDaq._TwSetDaqParameterBool.argtypes = [ct.c_char_p, ct.c_bool]
    return libTofDaq._TwSetDaqParameterBool(Parameter, Value)

def TwSetDaqParameterFloat(Parameter, Value):
    libTofDaq._TwSetDaqParameterFloat.argtypes = [ct.c_char_p, ct.c_float]
    return libTofDaq._TwSetDaqParameterFloat(Parameter, Value)

def TwSetDaqParameterInt64(Parameter, Value):
    libTofDaq._TwSetDaqParameterInt64.argtypes = [ct.c_char_p, ct.c_int64]
    return libTofDaq._TwSetDaqParameterInt64(Parameter, Value)

def TwSetDaqParameterDouble(Parameter, Value):
    libTofDaq._TwSetDaqParameterDouble.argtypes = [ct.c_char_p, ct.c_double]
    return libTofDaq._TwSetDaqParameterDouble(Parameter, Value)


def TwConfigVarNbrMemories(Enable, StepAtBuf, NbrMemoriesForStep):
    if len(StepAtBuf) != len(NbrMemoriesForStep):
        return TwError
    int_array_type = ct.c_int*len(StepAtBuf)
    Sarray = int_array_type(*StepAtBuf)
    Marray = int_array_type(*NbrMemoriesForStep)
    libTofDaq._TwConfigVarNbrMemories.argtypes = [ct.c_int, ct.c_int, int_array_type, int_array_type]
    return libTofDaq._TwConfigVarNbrMemories(Enable, len(StepAtBuf), Sarray, Marray)

def TwSetMassCalib(mode, nbrParams, p, nbrPoints, mass, tof, weight):
    libTofDaq._TwSetMassCalib.argtypes = [ct.c_int, ct.c_int, ndpointer(np.float64), ct.c_int, ndpointer(np.float64),  ndpointer(np.float64),  ndpointer(np.float64)]
    return libTofDaq._TwSetMassCalib(mode, nbrParams, p, nbrPoints, mass, tof, weight)


#---------------------------- DATA ACCESS FUNCTIONS ---------------------------- 

def TwGetDescriptor(ShMemDescriptor):
    libTofDaq._TwGetDescriptor.argtypes = [ct.POINTER(TSharedMemoryDesc)]
    return libTofDaq._TwGetDescriptor(ct.pointer(ShMemDescriptor))      

# If PeakPar is None, a new TPeakPar is allocated, filled and returned
def TwGetPeakParameters(PeakIndex, PeakPar=None):
    if PeakPar != None:
        return libTofDaq._TwGetPeakParameters(ct.pointer(PeakPar), PeakIndex)
    else:
        tempPeakPar = TPeakPar()
        tempRv = libTofDaq._TwGetPeakParameters(ct.pointer(tempPeakPar), PeakIndex)
        if tempRv == TwSuccess:
            return (tempPeakPar.label, tempPeakPar.mass, tempPeakPar.loMass, tempPeakPar.hiMass)
        else:
            return tempRv

def TwGetSharedMemory(ShMem, keepMapped):
    libTofDaq._TwGetSharedMemory.argtypes = [ct.POINTER(TSharedMemoryPointer), ct.c_bool]
    return libTofDaq._TwGetSharedMemory(ct.pointer(ShMem), keepMapped)

def TwReleaseSharedMemory():
    return libTofDaq._TwReleaseSharedMemory()

def TwWaitForNewData(Timeout, ShMemDescriptor, ShMem, WaitForEventReset):
    if ShMem == None:
        libTofDaq._TwWaitForNewData.argtypes = [ct.c_int, ct.c_void_p, ct.POINTER(TSharedMemoryPointer), ct.c_bool]
        return libTofDaq._TwWaitForNewData(Timeout, ct.pointer(ShMemDescriptor), None, WaitForEventReset)
    else:
        libTofDaq._TwWaitForNewData.argtypes = [ct.c_int, ct.POINTER(TSharedMemoryDesc), ct.POINTER(TSharedMemoryPointer), ct.c_bool]
        return libTofDaq._TwWaitForNewData(Timeout, ct.pointer(ShMemDescriptor), ct.pointer(ShMem), WaitForEventReset)

def TwWaitForEndOfAcquisition(timeout):
    libTofDaq._TwWaitForEndOfAcquisition.argtypes = [ct.c_int]
    return libTofDaq._TwWaitForEndOfAcquisition(timeout)

def TwGetMassCalib(mode, nbrParams, p, nbrPoints, mass, tof, weight):
    libTofDaq._TwGetMassCalib.argtypes = [ndpointer(np.int, shape=1),
                                                  ndpointer(np.int),
                                                  _double_array,
                                                  ndpointer(np.int, shape=1),
                                                  _double_array,
                                                  _double_array,
                                                  _double_array]
    return libTofDaq._TwGetMassCalib(mode, nbrParams, p, nbrPoints, mass, tof, weight)

def TwGetSumSpectrumFromShMem(spectrum, normalize):
    libTofDaq._TwGetSumSpectrumFromShMem.argtypes = [_double_array, ct.c_byte]
    return libTofDaq._TwGetSumSpectrumFromShMem(spectrum, normalize)

def TwGetTofSpectrumFromShMem(spectrum, segmentIndex, segmentEndIndex, bufIndex, normalize):
    libTofDaq._TwGetTofSpectrumFromShMem.argtypes = [_float_array, ct.c_int, ct.c_int, ct.c_int, ct.c_byte]
    return libTofDaq._TwGetTofSpectrumFromShMem(spectrum, segmentIndex, segmentEndIndex, bufIndex, normalize)

def TwGetSpecXaxisFromShMem(specAxis, axisType, unitLabel):
    libTofDaq._TwGetSpecXaxisFromShMem.argtypes = [_double_array, ct.c_int, ct.POINTER(ct.c_char)]
    return libTofDaq._TwGetSpecXaxisFromShMem(specAxis, axisType, unitLabel)

def TwGetStickSpectrumFromShMem(spectrum, masses, segmentIndex, segmentEndIndex, bufIndex):
    if masses == None:
        libTofDaq._TwGetStickSpectrumFromShMem.argtypes = [_float_array, ct.c_void_p, ct.c_int, ct.c_int, ct.c_int]
    else:
        libTofDaq._TwGetStickSpectrumFromShMem.argtypes = [_float_array, _float_array, ct.c_int, ct.c_int, ct.c_int]
    return libTofDaq._TwGetStickSpectrumFromShMem(spectrum, masses, segmentIndex, segmentEndIndex, bufIndex) 
    
def TwGetSegmentProfileFromShMem(segmentProfile, peakIndex, bufIndex):
    libTofDaq._TwGetSegmentProfileFromShMem.argtypes = [_float_array, ct.c_int, ct.c_int]
    return libTofDaq._TwGetSegmentProfileFromShMem(segmentProfile, peakIndex, bufIndex)

def TwGetBufTimeFromShMem(bufTime, bufIndex, writeIndex):
    libTofDaq._TwGetBufTimeFromShMem.argtypes = [_double_array, ct.c_int, ct.c_int]
    return libTofDaq._TwGetBufTimeFromShMem(bufTime, bufIndex, writeIndex)


# ------------------- DATA STORAGE FUNCTIONS -----------------------------------
def TwAddLogEntry(logEntry, logEntryTime):
    libTofDaq._TwAddLogEntry.argtypes = [ct.c_char_p, ct.c_uint64]
    return libTofDaq._TwAddLogEntry(logEntry, logEntryTime)

def TwAddAttributeInt(objName, attributeName, value):
    libTofDaq._TwAddAttributeInt.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_int]
    return libTofDaq._TwAddAttributeInt(objName, attributeName, value)

def TwAddAttributeDouble(objName, attributeName, value):
    libTofDaq._TwAddAttributeDouble.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_double]
    return libTofDaq._TwAddAttributeDouble(objName, attributeName, value)

def TwAddAttributeString(objName, attributeName, value):
    libTofDaq._TwAddAttributeString.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p]
    return libTofDaq._TwAddAttributeString(objName, attributeName, value)    

def TwAddUserData(location, nbrElements, elementDescription, data):
    libTofDaq._TwAddUserData.argtypes = [ct.c_char_p, ct.c_int, ct.c_char_p, _double_array]
    return libTofDaq._TwAddUserData(location, nbrElements, elementDescription, data)

def TwRegisterUserDataBuf(location, nbrElements, elementDescription, compressionLevel):
    libTofDaq._TwRegisterUserDataBuf.argtypes = [ct.c_char_p, ct.c_int, ct.c_char_p, ct.c_int]
    return libTofDaq._TwRegisterUserDataBuf(location, nbrElements, elementDescription, compressionLevel)

def TwRegisterUserDataBufPy(location, elementDescription, compressionLevel):
    if elementDescription != None:
        descBuffer = ct.create_string_buffer(len(elementDescription)*256)
        for index in range(len(elementDescription)):
            if isinstance(elementDescription[index], str):
                tempBuffer = ct.create_string_buffer(elementDescription[index].encode())
            else:
                tempBuffer = ct.create_string_buffer(elementDescription[index])
            ct.memmove(ct.addressof(descBuffer)+index*256, ct.addressof(tempBuffer), 256) 
    else:
        descBuffer = None
    if isinstance(location, str):
        location = location.encode()
    return TwRegisterUserDataBuf(location, len(elementDescription), descBuffer, compressionLevel)

def TwRegisterUserDataWrite(location, nbrElements, elementDescription, compressionLevel):
    libTofDaq._TwRegisterUserDataWrite.argtypes = [ct.c_char_p, ct.c_int, ct.c_char_p, ct.c_int]
    return libTofDaq._TwRegisterUserDataWrite(location, nbrElements, elementDescription, compressionLevel)

def TwRegisterUserDataWritePy(location, elementDescription, compressionLevel):
    if elementDescription != None:
        descBuffer = ct.create_string_buffer(len(elementDescription)*256)
        for index in range(len(elementDescription)):
            if isinstance(elementDescription[index], str):
                tempBuffer = ct.create_string_buffer(elementDescription[index].encode())
            else:
                tempBuffer = ct.create_string_buffer(elementDescription[index])
            ct.memmove(ct.addressof(descBuffer)+index*256, ct.addressof(tempBuffer), 256) 
    else:
        descBuffer = None
    if isinstance(location, str):
        location = location.encode()
    return TwRegisterUserDataWrite(location, len(elementDescription), descBuffer, compressionLevel)

def TwUnregisterUserData(location):
    libTofDaq._TwUnregisterUserData.argtypes = [ct.c_char_p]
    return libTofDaq._TwUnregisterUserData(location)

def TwUpdateUserData(location, nbrElements, data):
    libTofDaq._TwUpdateUserData.argtypes = [ct.c_char_p, ct.c_int, _double_array]
    return libTofDaq._TwUpdateUserData(location, nbrElements, data)

def TwUpdateUserDataPy(location, data):
    dataBuffer = (c_double*len(data))()
    for elementIndex in range(len(data)):
        dataBuffer[elementIndex] = data[elementIndex]
    if isinstance(location, str):
        location = location.encode()
    return TwUpdateUserData(location, len(data), dataBuffer)

    
def TwReadRegUserData(location, nbrElements, data):
    libTofDaq._TwReadRegUserData.argtypes = [ct.c_char_p, ct.c_int, _double_array]
    return libTofDaq._TwReadRegUserData(location, nbrElements, data)


def TwSetRegUserDataTarget(location, elementIndex, elementValue, blockTime):
    libTofDaq._TwSetRegUserDataTarget.argtypes = [ct.c_char_p, ct.c_int, ct.c_double, ct.c_int]
    return libTofDaq._TwSetRegUserDataTarget(location, elementIndex, elementValue, blockTime)

def TwGetRegUserDataTargetRange(location, elementIndex, minValue, maxValue):
    libTofDaq._TwGetRegUserDataTargetRange.argtypes = [ct.c_char_p, ct.c_int, ndpointer(np.float64, shape=1), ndpointer(np.float64, shape=1)]
    return libTofDaq._TwGetRegUserDataTargetRange(location, elementIndex, minValue, maxValue)
    

# ------------------------ TPS REMOTE CONTROL FUNCTIONS ------------------------

def TwTpsConnect():
    return libTofDaq._TwTpsConnect()
	
def TwTpsConnect2(ip, tpsType):
    libTofDaq._TwTpsConnect2.argtypes = [ct.c_char_p, ct.c_int]
    return libTofDaq._TwTpsConnect2(ip, tpsType)
    
def TwTpsDisconnect():
    return libTofDaq._TwTpsDisconnect()
    
def TwTpsGetMonitorValue(moduleCode, value):
    libTofDaq._TwTpsGetMonitorValue.argtypes = [ct.c_int, ndpointer(np.float64, shape=1)]
    return libTofDaq._TwTpsGetMonitorValue(moduleCode, value)

def TwTpsGetTargetValue(moduleCode, value):
    libTofDaq._TwTpsGetTargetValue.argtypes = [ct.c_int, ndpointer(np.float64, shape=1)]
    return libTofDaq._TwTpsGetTargetValue(moduleCode, value)

def TwTpsGetLastSetValue(moduleCode, value):
    libTofDaq._TwTpsGetLastSetValue.argtypes = [ct.c_int, ndpointer(np.float64, shape=1)]
    return libTofDaq._TwTpsGetLastSetValue(moduleCode, value)

def TwTpsSetTargetValue(moduleCode, value):
    libTofDaq._TwTpsSetTargetValue.argtypes = [ct.c_int, ct.c_double]
    return libTofDaq._TwTpsSetTargetValue(moduleCode, value)

def TwTpsGetNbrModules(nbrModules):
    libTofDaq._TwTpsGetNbrModules.argtypes = [ndpointer(np.int32, shape=1)]
    return libTofDaq._TwTpsGetNbrModules(nbrModules)

def TwTpsGetModuleCodes(moduleCodeBuffer, bufferLength):
    libTofDaq._TwTpsGetModuleCodes.argtypes = [ndpointer(np.int32), ct.c_int]
    return libTofDaq._TwTpsGetModuleCodes(moduleCodeBuffer, bufferLength)

##more pythonesque function, less error handling
def TwTpsGetModuleCodesPy():
    nbrModules = np.zeros((1,), dtype=np.int32)
    if TwTpsGetNbrModules(nbrModules) != 4:
        return []
    moduleCodes = np.zeros(nbrModules, dtype=np.int32)
    if TwTpsGetModuleCodes(moduleCodes, len(moduleCodes)) != 4:
        return []
    return moduleCodes.astype(int).tolist()    

def TwTpsInitialize():
    libTofDaq._TwTpsInitialize.argtypes = []
    return libTofDaq._TwTpsInitialize()    

def TwTpsSetAllVoltages():
    return libTofDaq._TwTpsSetAllVoltages()
    
def TwTpsShutdown():
    libTofDaq._TwTpsShutdown.argtypes = []
    return libTofDaq._TwTpsShutdown()

def TwTpsGetStatus(status):
    libTofDaq._TwTpsGetStatus.argtypes = [ndpointer(np.int32, shape=1)]
    return libTofDaq._TwTpsGetStatus(status)

def TwTpsLoadSetFile(setFile):
    libTofDaq._TwTpsLoadSetFile.argtypes = [ct.c_char_p]
    return libTofDaq._TwTpsLoadSetFile(setFile)

def TwTpsSaveSetFile(setFile):
    libTofDaq._TwTpsSaveSetFile.argtypes = [ct.c_char_p]
    return libTofDaq._TwTpsSaveSetFile(setFile)

if TOFDAQDLL_REV >= 1083:
    def TwTpsGetModuleProperties(moduleCode, properties, label):
        libTofDaq._TwTpsGetModuleProperties.argtypes = [ct.c_int, ndpointer(np.int32, shape=1), ct.c_char_p]
        return libTofDaq._TwTpsGetModuleProperties(moduleCode, properties, label)

    def TwTpsGetModulePropertiesPy(moduleCode):
        prop = np.ndarray((1,), dtype=np.int32)
        label = b'\0'*256
        if (TwTpsGetModuleProperties(moduleCode, prop, label) == TwSuccess):
            labelStr = label.split(b'\0', 1)[0].decode()
            properties = int(prop[0]) 
            return {'label': labelStr, 'hasMonitor': (properties&0x1 != 0), 'isSettable': (properties&0x2 != 0), 'isTrigger': (properties&0x4 != 0)}
        else:
            return {}
        

if TOFDAQDLL_REV >= 1120:
    def TwTpsSendPdo(cobId, data):
        libTofDaq._TwTpsSendPdo.argtypes = [ct.c_int, ct.c_int, ct.c_char_p]
        return libTofDaq._TwTpsSendPdo(cobId, len(data), data)


if TOFDAQDLL_REV >= 1204:
    def TwSendDioStartSignal():
        return libTofDaq._TwSendDioStartSignal()

    def TwWaitingForDioStartSignal():
        libTofDaq._TwWaitingForDioStartSignal.restype = ct.c_bool
        return libTofDaq._TwWaitingForDioStartSignal()
        
def TwSetTimeout(timeout):
    libTofDaq._TwSetTimeout.argtypes = [ct.c_int]
    libTofDaq._TwSetTimeout(timeout)
    

