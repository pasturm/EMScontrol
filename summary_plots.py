import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import ctypes as ct
from TwH5 import *


def read_energyData(file):
    bufLength = np.zeros((1,), dtype=ct.c_int32)
    rv = TwGetRegUserDataFromH5(file, '/EnergyData'.encode(), -1, -1, bufLength, None, None)
    # get description length
    descLength = np.zeros((1,), dtype=ct.c_int32)
    rv = TwGetRegUserDataFromH5(file, '/EnergyData'.encode(), 0, 0, descLength, None, None)
    # get user data and description
    desc_str = ct.create_string_buffer(int(descLength[0]*256))
    dataBuffer = np.ndarray(bufLength, dtype=np.float64)
    rv = TwGetRegUserDataFromH5(file, '/EnergyData'.encode(), -1, -1, bufLength, dataBuffer, desc_str)
    description = []
    for i in range(descLength[0]):
        description.append((desc_str[i*256:(i+1)*256].strip(b'\0')).decode())
    energyData = np.reshape(np.array(dataBuffer), [-1,descLength[0]])
    return energyData


def read_peakData(file, m):
    desc = TwH5Desc()
    TwGetH5Descriptor(file, desc)
    peakParType = TPeakPar * desc.nbrPeaks
    peakPar = peakParType()
    peakIndex = -1
    TwGetPeakParametersFromH5(file, peakPar, peakIndex)
    label = np.empty(desc.nbrPeaks, dtype=object)
    mass = np.zeros(desc.nbrPeaks)
    loMass = np.zeros(desc.nbrPeaks)
    hiMass = np.zeros(desc.nbrPeaks)
    for i in range(desc.nbrPeaks):
        label[i] = peakPar[i].label.decode()
        mass[i] = peakPar[i].mass
        loMass[i] = peakPar[i].loMass
        hiMass[i] = peakPar[i].hiMass
    PeakTable = pd.DataFrame({'label':label, 'mass':mass, 'loMass':loMass, 'hiMass':hiMass})
    peakIndex = PeakTable.index.values[PeakTable['mass']==m]
    peakOffset = int(peakIndex)
    peakCount = 1
    segOffset = 0
    segCount = desc.nbrSegments
    bufOffset = 0
    bufCount =desc.nbrBufs
    writeOffset = 0
    writeCount = desc.nbrWrites
    bufferSize = peakCount * segCount * bufCount * writeCount
    dataBuffer = np.zeros(bufferSize, dtype=np.float32)
    TwGetPeakData(file, peakOffset, peakCount, segOffset, segCount, bufOffset,  bufCount,  writeOffset,  writeCount, dataBuffer)
    TwCloseH5(file)
    data = dataBuffer.reshape(-1, segCount)
    PeakData = pd.DataFrame(data)
    return PeakData


def plot_plotly_avgSegProfile(file, m, log):
    desc = TwH5Desc()
    TwGetH5Descriptor(file, desc)
    PeakData = read_peakData(file, m)
    fig = px.line(x=np.arange(desc.nbrSegments)*desc.tofPeriod*1e6, y=PeakData.mean(axis=0), 
        title=f'Energy-averaged segment profile, m/Q = {m} Th', labels=dict(x="time (&mu;s)", y="ion/extraction"), log_y=log, template='plotly_dark')
    fig.add_annotation(x=1, y=-0.1, text=f'{file.decode()}', xref="paper", yref="paper", showarrow=False, align='right')
    return fig


def plot_plotly_iedf(file, m, log):
    energyData = read_energyData(file)
    PeakData = read_peakData(file, m)
    fig = px.line(x=energyData[:,0], y=PeakData.mean(axis=1), 
        title=f'IEDF, m/Q = {m} Th', labels=dict(x="ion energy (eV)", y="ion/extraction"), log_y=log, template='plotly_dark')
    fig.add_annotation(x=1, y=-0.1, text=f'{file.decode()}', xref="paper", yref="paper", showarrow=False, align='right')
    return fig


def plot_plotly_heatmap(file, m, log):
    desc = TwH5Desc()
    TwGetH5Descriptor(file, desc)
    energyData = read_energyData(file)
    PeakData = read_peakData(file, m)
    if log:
        with np.errstate(divide='ignore'):
            zdata = np.log(PeakData)
        zlabel = 'log(ions/extraction)'
    else:
        zdata = PeakData
        zlabel = 'ions/extraction'
    fig = px.imshow(zdata, color_continuous_scale='Viridis', origin='lower', aspect="auto",
        labels=dict(x="time (&mu;s)", y="ion energy (eV)", color=zlabel),
        x=np.arange(desc.nbrSegments)*desc.tofPeriod*1e6, y=energyData[:,0], title=f'm/Q = {m} Th', template='plotly_dark'
    )
    fig.add_annotation(x=1, y=-0.1, text=f'{file.decode()}', xref="paper", yref="paper", showarrow=False, align='right')
    return fig


def plot_plotly_3dsurface(file, m, log):
    desc = TwH5Desc()
    TwGetH5Descriptor(file, desc)
    energyData = read_energyData(file)
    PeakData = read_peakData(file, m)
    if log:
        with np.errstate(divide='ignore'):
            zdata = np.log(PeakData)
        zlabel = 'log(ions/extraction)'
    else:
        zdata = PeakData
        zlabel = 'ions/extraction'
    fig = go.Figure(data=[go.Surface(z=zdata, x=np.arange(desc.nbrSegments)*desc.tofPeriod*1e6, y=energyData[:,0], 
        colorscale = 'Viridis',
        opacity=0.8)])
    fig.update_layout(
        title=f'm/Q = {m} Th', 
        scene = dict(xaxis_title='time (\u03BCs)', yaxis_title='ion energy (eV)', zaxis_title=zlabel),
        template='plotly_dark')
        # template='plotly_white')
    fig.add_annotation(x=1, y=-0.1, text=f'{file.decode()}', xref="paper", yref="paper", showarrow=False, align='right')
    return fig
