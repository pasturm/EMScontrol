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


libname = {'linux':'libtwtool.so', 'linux2':'libtwtool.so', 'darwin':'libtwtool.dylib', 'win32':'TwToolDll.dll'}
toollib = ct.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), libname[sys.platform]))

tof2mass = toollib.TwTof2Mass if os.name=='posix' else toollib._TwTof2Mass
def TwTof2Mass(tofSample, massCalibMode, p):
    tof2mass.restype = ct.c_double
    if isinstance(p, np.ndarray):
        tof2mass.argtypes = [ct.c_double, ct.c_int, ndpointer(np.float64)]
    else:
        tof2mass.argtypes = [ct.c_double, ct.c_int, ct.POINTER(ct.c_double)]
    return tof2mass(tofSample, massCalibMode, p)

mass2tof = toollib.TwMass2Tof if os.name=='posix' else toollib._TwMass2Tof
def TwMass2Tof(mass, massCalibMode, p):
    mass2tof.restype = ct.c_double
    if isinstance(p, np.ndarray):
        mass2tof.argtypes = [ct.c_double, ct.c_int, ndpointer(np.float64)]
    else:
        mass2tof.argtypes = [ct.c_double, ct.c_int, ct.POINTER(ct.c_double)]
    return mass2tof(mass, massCalibMode, p)

translaterv = toollib.TwTranslateReturnValue if os.name=='posix' else toollib._TwTranslateReturnValue
def TwTranslateReturnValue(ReturnValue):
    translaterv.argtypes = [ct.c_int]
    translaterv.restype = ct.c_char_p
    return translaterv(ReturnValue)

fitsinglepeak = toollib.TwFitSinglePeak if os.name=='posix' else toollib._TwFitSinglePeak
def TwFitSinglePeak(nbrDataPoints, yVals, xVals, peakType, blOffset, blSlope, amplitude, fwhmLo, fwhmHi, peakPos, mu):    
    if isinstance(yVals, np.ndarray):
        fitsinglepeak.argtypes = [ct.c_int, ndpointer(np.float64, shape=nbrDataPoints), ndpointer(np.float64, shape=nbrDataPoints), ct.c_int, ndpointer(np.float64, shape=1), ndpointer(np.float64, shape=1), ndpointer(np.float64, shape=1), ndpointer(np.float64, shape=1), ndpointer(np.float64, shape=1), ndpointer(np.float64, shape=1), ndpointer(np.float64, shape=1)]
    else:
        fitsinglepeak.argtypes = [ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
    return fitsinglepeak(nbrDataPoints, yVals, xVals, peakType, blOffset, blSlope, amplitude, fwhmLo, fwhmHi, peakPos, mu)

fitsinglepeak2 = toollib.TwFitSinglePeak2 if os.name=='posix' else toollib._TwFitSinglePeak2
def TwFitSinglePeak2(nbrDataPoints, yVals, xVals, peakType, param):
    if isinstance(yVals, np.ndarray):
        fitsinglepeak2.argtypes = [ct.c_int, ndpointer(np.float64, shape=nbrDataPoints), ndpointer(np.float64, shape=nbrDataPoints), ct.c_int,  ndpointer(np.float64, shape=7)]   
    else:
        fitsinglepeak2.argtypes = [ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int, ct.POINTER(ct.c_double)]   
    return fitsinglepeak2(nbrDataPoints, yVals, xVals, peakType, param)

evalsinglepeak = toollib.TwEvalSinglePeak if os.name=='posix' else toollib._TwEvalSinglePeak
def TwEvalSinglePeak(xVal, param):
    evalsinglepeak.restype = ct.c_double
    if isinstance(param, np.ndarray):
        evalsinglepeak.argtypes = [ct.c_double, ndpointer(np.float64, shape=7)]
    else:
        evalsinglepeak.argtypes = [ct.c_double, ct.POINTER(ct.c_double)]   
    return evalsinglepeak(xVal, param)

getmoleculemass = toollib.TwGetMoleculeMass if os.name=='posix' else toollib._TwGetMoleculeMass
def TwGetMoleculeMass(molecule, mass):
    if isinstance(mass, np.ndarray):
        getmoleculemass.argtypes = [ct.c_char_p, ndpointer(np.float64, shape=1)]
    else:
        getmoleculemass.argtypes = [ct.c_char_p, ct.POINTER(ct.c_double)]   
    return getmoleculemass(molecule, mass)

multipeakfit = toollib.TwMultiPeakFit if os.name=='posix' else toollib._TwMultiPeakFit
def TwMultiPeakFit(nbrDataPoints, dataX, dataY, nbrPeaks, mass, intensity, commonPar, options):
    if isinstance(dataX, np.ndarray):
        multipeakfit.argtypes = [ct.c_int, ndpointer(np.float64, shape=nbrDataPoints), ndpointer(np.float64, shape=nbrDataPoints), ct.c_int, ndpointer(np.float64, shape=nbrPeaks), ndpointer(np.float64, shape=nbrPeaks), ndpointer(np.float64, shape=6), ct.c_int]
    else:
        multipeakfit.argtypes = [ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int]
    return multipeakfit(nbrDataPoints, dataX, dataY, nbrPeaks, mass, intensity, commonPar, options)

evalmultipeak = toollib.TwEvalMultiPeak if os.name=='posix' else toollib._TwEvalMultiPeak
def TwEvalMultiPeak(x, nbrPeaks, mass, intensity, commonPar):
    evalmultipeak.restype = ct.c_double
    if isinstance(mass, np.ndarray):
        evalmultipeak.argtypes = [ct.c_double, ct.c_int, ndpointer(np.float64, shape=nbrPeaks), ndpointer(np.float64, shape=nbrPeaks), ndpointer(np.float64, shape=6)]
    else:
        evalmultipeak.argtypes = [ct.c_double, ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]   
    return evalmultipeak(x, nbrPeaks, mass, intensity, commonPar)

fitresolution = toollib.TwFitResolution if os.name=='posix' else toollib._TwFitResolution
def TwFitResolution(nbrPoints, mass, resolution, R0, m0, dm):
    if isinstance(mass, np.ndarray):
        fitresolution.argtypes = [ct.c_int, ndpointer(np.float64, shape=nbrPoints), ndpointer(np.float64, shape=nbrPoints), ndpointer(np.float64, shape=1), ndpointer(np.float64, shape=1), ndpointer(np.float64, shape=1)]
    else:
        fitresolution.argtypes = [ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
    return fitresolution(nbrPoints, mass, resolution, R0, m0, dm)

evalresolution = toollib.TwEvalResolution if os.name=='posix' else toollib._TwEvalResolution
def TwEvalResolution(R0, m0, dm, mass):
    evalresolution.restype = ct.c_double
    evalresolution.argtypes = [ct.c_double, ct.c_double, ct.c_double, ct.c_double]
    return evalresolution(R0, m0, dm, mass)
    
matchspectra = toollib.TwMatchSpectra if os.name=='posix' else toollib._TwMatchSpectra
def TwMatchSpectra(spec1, spec2, nbrPoints, matchMethod, matchScore=None):
    matchspectra.argtypes = [ndpointer(np.float64, shape=nbrPoints), ndpointer(np.float64, shape=nbrPoints), ct.c_int, ct.c_int, ndpointer(np.float64, shape=1)]
    if matchScore is None:
        tempScore = np.ndarray((1,), dtype=np.float64)
        rv = matchspectra(spec1, spec2, nbrPoints, matchMethod, tempScore)
        if rv == 4:
            return tempScore[0]
        else:
            return 0.0
    else:
        return matchspectra(spec1, spec2, nbrPoints, matchMethod, matchScore)

makemqaxis = toollib.TwMakeMqAxis if os.name=='posix' else toollib._TwMakeMqAxis
def TwMakeMqAxis(mqAxis, massCalibMode, p):
    if isinstance(p, np.ndarray):
        makemqaxis.argtypes = [ndpointer(np.float64), ct.c_int, ct.c_int, ndpointer(np.float64)]
    else:
        makemqaxis.argtypes = [ndpointer(np.float64), ct.c_int, ct.c_int, ct.POINTER(ct.c_double)]
    return makemqaxis(mqAxis, mqAxis.shape[0], massCalibMode, p)


getisotopepattern = toollib.TwGetIsotopePattern if os.name=='posix' else toollib._TwGetIsotopePattern
def TwGetIsotopePattern(molecule, abundanceLimit, nbrIsotopes, isoMass, isoAbundance):
    getisotopepattern.argtypes = [ct.c_char_p, ct.c_double]
    if isinstance(nbrIsotopes, np.ndarray):
        getisotopepattern.argtypes += [ndpointer(np.int32, shape=1)]
    else:
        getisotopepattern.argtypes += [ct.POINTER(ct.c_int)]
    if (isoMass is None) and (isoAbundance is None):
        getisotopepattern.argtypes += [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
    else:
        getisotopepattern.argtypes += [ndpointer(np.float64), ndpointer(np.float64)]
    return getisotopepattern(molecule, abundanceLimit, ct.byref(nbrIsotopes), isoMass, isoAbundance)
        
def TwGetIsotopePatternPy(molecule, abundanceLimit):
    nIso = ct.c_int(0)
    if isinstance(molecule, str):
        molecule = molecule.encode()
    retVal = TwGetIsotopePattern(molecule, abundanceLimit, nIso, None, None)
    if (retVal == 9):
        isoMass = np.ndarray((nIso.value,), dtype=np.float64)
        isoAb = np.ndarray((nIso.value,), dtype=np.float64)
        if (TwGetIsotopePattern(molecule, abundanceLimit, nIso, isoMass, isoAb) == 4):
            return (isoMass, isoAb)
    else:
        print(retVal)
        return None;
    
encimscorrelateprofile = toollib.TwEncImsCorrelateProfile if os.name=='posix' else toollib._TwEncImsCorrelateProfile
def TwEncImsCorrelateProfile(profile, opMode, par):
    encimscorrelateprofile.argtypes = [ndpointer(np.float32), ct.c_int, ndpointer(np.int32)]
    return encimscorrelateprofile(profile, opMode, par)


matchspectra = toollib.TwMatchSpectra if os.name=='posix' else toollib._TwMatchSpectra
def TwMatchSpectra(spec1, spec2, nbrPoints, matchMethod, matchScore):
    matchspectra.argtypes = [ndpointer(np.float64, shape=nbrPoints), ndpointer(np.float64, shape=nbrPoints), ct.c_int, ct.c_int, ndpointer(np.float64, shape=1)]
    return matchspectra(spec1, spec2, nbrPoints, matchMethod, matchScore)


integratetofspectrum = toollib.TwIntegrateTofSpectrum if os.name=='posix' else toollib._TwIntegrateTofSpectrum
def TwIntegrateTofSpectrum(tofSpec, scaleFactor, mcMode, mcPar, peak, stickSpec, algorithm):
    integratetofspectrum.argtypes = [ndpointer(np.float32), ct.c_int, ct.c_float, ct.c_int, ndpointer(np.float64), ct.c_int, ndpointer(dtype=TPeakPar), ndpointer(np.float32), ct.c_int, ct.POINTER(ct.c_double)]
    return integratetofspectrum(tofSpec, tofSpec.shape[0], scaleFactor, mcMode, mcPar, peak.shape[0], peak, stickSpec, algorithm, None)


integratetofspectra = toollib.TwIntegrateTofSpectra if os.name=='posix' else toollib._TwIntegrateTofSpectra
floatPtr = ct.POINTER(ct.c_float)
integratetofspectra.argtypes = [ct.POINTER(floatPtr), ct.c_int, ct.c_int, ct.c_float, ct.c_int, ndpointer(np.float64), ct.c_int, ndpointer(dtype=TPeakPar), ct.POINTER(floatPtr), ct.c_int, ct.POINTER(ct.c_double)]
def TwIntegrateTofSpectra(tofSpecs, scaleFactor, mcMode, mcPar, peak, stickSpecs, algorithm):
    if not isinstance(tofSpecs, np.ndarray) or not isinstance(stickSpecs, np.ndarray) or len(tofSpecs.shape) != 2 or len(stickSpecs.shape) != 2:
        raise TypeError('tofSpecs and stickSpecs must be 2D numpy arrays')        
    nbrSpec = len(tofSpecs)
    nbrSamples = len(tofSpecs[0])
    nbrPeaks = len(peak)
    #sanity check stickSpecs dimensions (not possible using shape in ndpointer in argtypes as it is passed as naked ctypes pointer)
    if stickSpecs.shape != (nbrSpec, nbrPeaks):
        raise TypeError('stickSpecs array dimension missmatch')
    #could not figure out a way to pass 2D numpy arrays to the function without explicitely creating the ctypes array of pointers:    
    tofPtrPtr = (floatPtr * nbrSpec)(*[t.ctypes.data_as(floatPtr) for t in tofSpecs])
    stickPtrPtr = (floatPtr * nbrSpec)(*[s.ctypes.data_as(floatPtr) for s in stickSpecs])    
    return integratetofspectra(tofPtrPtr, nbrSamples, nbrSpec, scaleFactor, mcMode, mcPar, nbrPeaks, peak, stickPtrPtr, algorithm, None)


decomposemass = toollib.TwDecomposeMass if os.name=='posix' else toollib._TwDecomposeMass
def TwDecomposeMass(targetMass, tolerance, nbrAtoms, atomMass, atomLabel, nbrFilters, elementIndex1, elementIndex2, filterMinVal, filterMaxVal, nbrCompomers):
    if nbrFilters > 0:
        decomposemass.argtypes = [ct.c_double, ct.c_double, ct.c_int, ndpointer(np.float64, shape=nbrAtoms), ct.c_char_p, ct.c_int, ndpointer(np.int32, shape=nbrFilters), ndpointer(np.int32, shape=nbrFilters), ndpointer(np.float64, shape=nbrFilters), ndpointer(np.float64, shape=nbrFilters), ndpointer(np.int32, shape=1)]
    else:
        decomposemass.argtypes = [ct.c_double, ct.c_double, ct.c_int, ndpointer(np.float64, shape=nbrAtoms), ct.c_char_p, ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ndpointer(np.int32, shape=1)]
    return decomposemass(targetMass, tolerance, nbrAtoms, atomMass, atomLabel, nbrFilters, elementIndex1, elementIndex2, filterMinVal, filterMaxVal, nbrCompomers)


decomposemass2 = toollib.TwDecomposeMass2 if os.name=='posix' else toollib._TwDecomposeMass2
def TwDecomposeMass2(targetMass, tolerance, nbrAtoms, atomMass, atomLabel, nbrFilters, elementIndex1, elementIndex2, filterMinVal, filterMaxVal, nbrCompomers, maxHits, maxSearch):
    if nbrFilters > 0:
        decomposemass2.argtypes = [ct.c_double, ct.c_double, ct.c_int, ndpointer(np.float64, shape=nbrAtoms), ct.c_char_p, ct.c_int, ndpointer(np.int32, shape=nbrFilters), ndpointer(np.int32, shape=nbrFilters), ndpointer(np.float64, shape=nbrFilters), ndpointer(np.float64, shape=nbrFilters), ndpointer(np.int32, shape=1), ct.c_int, ct.c_int]
    else:
        decomposemass2.argtypes = [ct.c_double, ct.c_double, ct.c_int, ndpointer(np.float64, shape=nbrAtoms), ct.c_char_p, ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ndpointer(np.int32, shape=1), ct.c_int, ct.c_int]
    return decomposemass2(targetMass, tolerance, nbrAtoms, atomMass, atomLabel, nbrFilters, elementIndex1, elementIndex2, filterMinVal, filterMaxVal, nbrCompomers, maxHits, maxSearch)


getcomposition = toollib.TwGetComposition if os.name=='posix' else toollib._TwGetComposition
getcomposition.argtypes = [ct.c_int, ct.c_char_p, ndpointer(np.int32, shape=1), ndpointer(np.float64, shape=1), ndpointer(np.float64, shape=1)]
def TwGetComposition(index, sumFormula, sumFormulaLength, mass, massError):
    return getcomposition(index, sumFormula, sumFormulaLength, mass, massError)


def TwDecomposeMassPy(targetMass, tolerance, fragments, filters=None, maxHits=9999999, maxSearch=9999999):
    """Finds sum formulae for a given molecular mass

    Parameters:
    targetMass (float): target mass for molecular formula search
    tolerance (float): maximum allowed mass error for result formulae
    fragments (list of tuples(float, bytes)): list of fragment masses and their string (bytes) representation
    filters (list of tuples(int, int, float, float)): 2 types of filter supported: absolute number for a given element/isotope (set 2nd element index to -1) and ratios between two defined elements/isotopes
    maxHits (int): abort when this many compomers have been found
    maxSearch (int): abort after that many tested formulas

    Returns:
    list of tuples(bytes, float, float): sum formula, mass and mass error

   """
    fragMass, fragLabel = zip(*fragments)
    atomLabel = b'\0'.join(fragLabel)
    atomLabel += b'\0'
    atomMass = np.array(fragMass, dtype=np.float64)
    nbrCompomers = np.zeros((1,), dtype=np.int32)
    if filters:
        elIndex1, elIndex2, minVal, maxVal = zip(*filters)
        elementIndex1 = np.array(elIndex1, dtype=np.int32)
        elementIndex2 = np.array(elIndex2, dtype=np.int32)
        filterMinVal = np.array(minVal, dtype=np.float64)
        filterMaxVal = np.array(maxVal, dtype=np.float64)        
        rv = TwDecomposeMass2(targetMass, tolerance, len(fragments), atomMass, atomLabel, len(filters), elementIndex1, elementIndex2, filterMinVal, filterMaxVal, nbrCompomers, maxHits, maxSearch)
    else:
        rv = TwDecomposeMass2(targetMass, tolerance, len(fragments), atomMass, atomLabel, 0, None, None, None, None, nbrCompomers, maxHits, maxSearch)
    result = []
    if (rv == TwSuccess or rv == TwAborted) and nbrCompomers[0] >= 0:
        formulaBuffer = ct.create_string_buffer(1024)
        formulaBufferLength = np.array([1024], dtype=np.int32)
        mass = np.zeros((1,), dtype=np.float64)
        massErr = np.zeros((1,), dtype=np.float64)
        for c in range(nbrCompomers[0]):
            TwGetComposition(c, formulaBuffer, formulaBufferLength, mass, massErr)
            result.append((formulaBuffer.value, mass[0], massErr[0]))
    return (result, rv == TwSuccess)


masscalibrate = toollib.TwMassCalibrate if os.name=='posix' else toollib._TwMassCalibrate
def TwMassCalibrate(mode, nbrPoints, mass, tof, weight, nbrParams, p):
    masscalibrate.argtypes = [ct.c_int, ct.c_int, ndpointer(np.float64, shape=nbrPoints), ndpointer(np.float64, shape=nbrPoints), ndpointer(np.float64, shape=nbrPoints), ndpointer(np.int32, shape=1), ndpointer(np.float64), ct.c_void_p, ct.c_void_p]
    return masscalibrate(mode, nbrPoints, mass, tof, weight, nbrParams, p, None, None)
    
    


    

