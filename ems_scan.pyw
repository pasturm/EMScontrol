#! /usr/bin/env python3

"""EMS voltage control and energy scanning"""

__version__ = '0.14.5'
__author__ = 'Patrick Sturm'
__copyright__ = 'Copyright 2021 TOFWERK'

import numpy as np
import pandas as pd
import time
import sys
import math
import logging
import ctypes
import threading
import os
import json
import copy
import re
import PySimpleGUI as sg
from TofDaq import *
from TwTool import *


# developer mode for offline testing
devmode = False  
# devmode = True  

# user settings
settings = sg.UserSettings(path='.')

# Logger
log = logging.getLogger(__name__)

# output logging messages to DebugView via OutputDebugString
OutputDebugString = ctypes.windll.kernel32.OutputDebugStringW
class DebugViewHandler(logging.Handler):
    def emit(self, record):
        OutputDebugString(self.format(record))

# TPS RC codes
tps1rc = {'L2': 14, 'DEFL': 15, 'DEFLFL': 16, 'IONEX': 17, 'L1': 18, 'REFERENCE': 117, 
    'ORIFICE': 2500, 'INNER_CYL': 2501, 'OUTER_CYL': 2502, 'MATSUDA': 2503, 
    'DEFL1U': 2504, 'DEFL1D': 2505, 'DEFL1L': 2506, 'DEFL1R': 2507, 'TOFREF': 202,
    'TOFEXTR1': 201, 'TOFEXTR2': 200, 'TOFPULSE': 203, 'RG': 2, 'RB': 1,
    'DRIFT': 9, 'PA': 5, 'MCP': 6, 'HVSUPPLY': 602, 'HVPOS': 603, 'HVNEG': 604}

# Window element keys that can be saved to a file
SETPOINTS = {'-ORIFICE-':0, '-LENS1-':0, '-DEFL1U-':0, '-DEFL1D-':0, '-DEFL1L-':0, '-DEFL1R-':0, 
    '-MATSUDA-':0, '-LENS2-':0, '-DEFL-':0, '-DEFLFL-':0, '-REF-':0, '-TOFEXTR1-':0, '-RG-':0,
    '-RB-':0, '-TOFEXTR2-':0, '-TOFPULSE-':0, '-DRIFT-':0, '-PA-':0, '-MCP-':0, '-IONEX-':0,
    '-ESA_ENERGY-':0, '-TOF_ENERGY-':0, '-ION_ENERGY-':0, '-HVSUPPLY-':0, '-HVPOS-':0, '-HVNEG-':0}

# exit event to abort energy scanning
exit_event = threading.Event()

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def is_all_numeric(values):
    """Check if all input strings are numeric"""
    for key in SETPOINTS:
        try:
            float(values[key])
        except ValueError:
            log.error((f'{key} string cannot be converted to numeric value.'))
            return False
    return True

def calculate_ESA_voltages(esa_energy, polarity, r0 = 0.100, d = 0.0125):
    """
    Calculates the cylinder electrode potentials for a given energy.
    
    Arguments:
    esa_energy: mean kinetic energy of singly charged ions, eV
    polarity: ion polarity, 1 for positive ion mode, -1 for negative ion mode
    r0: mean cylinder radius, m
    d: half-distance between inner and outer cylinder, m
    
    Return values:
    V1, V2: inner and outer cylinder voltage, V

    from Yavor, Optics of Charged Particle Analyzers.
    """
    r1 = r0-d  # inner cylinder radius, m
    r2 = r0+d  # outer cylinder radius, m
    V1 = polarity*esa_energy*2*math.log(r2/r1)*(math.log(r1)-math.log(r0))/(math.log(r2)-math.log(r1))  # inner cylinder voltage, V
    V2 = polarity*esa_energy*2*math.log(r2/r1) + V1  # outer cylinder voltage, V
    return V1, V2

def calculate_energies_from_ESA_voltages(V_inner, V_outer, polarity):
    """
    Calculates the ESA energy and starting ion energy from the cylinder electrode potentials.
    
    Arguments:
    V_inner: inner cylinder potential, V
    V_outer: outer cylinder potential, V
    polarity: ion polarity, 1 for positive ion mode, -1 for negative ion mode
    
    Return values:
    esa_energy, ion_energy : ESA and starting ion energy, eV
    """
    r0 = 0.100
    d = 0.0125
    r1 = r0-d  # inner cylinder radius, m
    r2 = r0+d  # outer cylinder radius, m
    c1 = 2*math.log(r2/r1)
    c2 = c1*(math.log(r1)-math.log(r0))/(math.log(r2)-math.log(r1))
    esa_energy = polarity*(V_outer-V_inner)/c1
    ion_energy = (polarity*V_inner*(1+(2*c2+c1-2)/c1)+polarity*V_outer*(1-(2*c2+c1-2)/c1))/2
    return esa_energy, ion_energy

def tps_error_log(rv, key):
    if (rv != TwSuccess and not devmode): log.error(f'Failed to set {key} voltage: {TwTranslateReturnValue(rv).decode()}.')

def calculate_energy_offset(esa_energy):
    """ from calibration with EI source """
    slope = -0.01265
    intercept = -0.955
    return slope*esa_energy+intercept

def get_ionmode():
    """ Checks TPS ion mode: 0=Inconsistent, 1=Undetermined, 2=Positive, 3=Negative """
    status = np.zeros(1, dtype=np.int32)
    TwTpsGetStatus(status)
    if (not devmode):
        ionmode = (int(status) & 0b01100000) >> 5
        if (ionmode==2):
            return 1
        elif (ionmode==3):
            return -1
        else:
            log.error('Ion mode undefined.')
            return 0
    else:
        return 1

def set_voltages_ea(values, ion_energy, polarity):
    """
    Set all ion_energy-dependent voltages.
    """
    V1, V2 = calculate_ESA_voltages(float(values['-ESA_ENERGY-']), polarity)
    energy_offset = calculate_energy_offset(float(values['-ESA_ENERGY-']))
    V_extractor = (ion_energy + energy_offset - float(values['-ESA_ENERGY-']))*polarity
    V_reference = float(values['-REF-'])
    V_tofreference = (ion_energy + energy_offset - float(values['-TOF_ENERGY-']))*polarity  # from LV channel -> with sign
    V_tofextractor1 = V_tofreference + float(values['-TOFEXTR1-'])  # from LV channel -> with sign, relative to TOF reference
    rg_correction = 0.25*polarity  # ion energy correction of RG in V/eV
    V_rg = float(values['-RG-']) + V_tofreference*rg_correction  # -RG- is set value if TOFREF = 0 V (ion energy - tof energy = 0 eV)
    V_lens1 = V_extractor + float(values['-LENS1-']) + 0.955*(float(values['-ESA_ENERGY-'])-100)*polarity
    V_matsuda = V_extractor + float(values['-MATSUDA-']) + 0.24*(float(values['-ESA_ENERGY-'])-100)*polarity
    tps_error_log(TwTpsSetTargetValue(tps1rc['ORIFICE'], float(values['-ORIFICE-'])), 'ORIFICE')
    tps_error_log(TwTpsSetTargetValue(tps1rc['IONEX'], V_extractor + float(values['-IONEX-'])), 'IONEX')
    tps_error_log(TwTpsSetTargetValue(tps1rc['L1'], V_lens1), 'L1')
    tps_error_log(TwTpsSetTargetValue(tps1rc['DEFL1U'], V_extractor + float(values['-DEFL1U-'])), 'DEFL1U')
    tps_error_log(TwTpsSetTargetValue(tps1rc['DEFL1D'], V_extractor + float(values['-DEFL1D-'])), 'DEFL1D')
    tps_error_log(TwTpsSetTargetValue(tps1rc['DEFL1R'], V_extractor + float(values['-DEFL1R-'])), 'DEFL1R')
    tps_error_log(TwTpsSetTargetValue(tps1rc['DEFL1L'], V_extractor + float(values['-DEFL1L-'])), 'DEFL1L')
    tps_error_log(TwTpsSetTargetValue(tps1rc['INNER_CYL'], V_extractor + V1), 'INNER_CYL')
    tps_error_log(TwTpsSetTargetValue(tps1rc['OUTER_CYL'], V_extractor + V2), 'OUTER_CYL')
    tps_error_log(TwTpsSetTargetValue(tps1rc['MATSUDA'], V_matsuda), 'MATSUDA')
    tps_error_log(TwTpsSetTargetValue(tps1rc['REFERENCE'], V_tofreference + V_reference), 'REFERENCE')
    tps_error_log(TwTpsSetTargetValue(tps1rc['L2'], V_tofreference + V_reference + float(values['-LENS2-'])), 'L2')
    tps_error_log(TwTpsSetTargetValue(tps1rc['DEFL'], V_tofreference + V_reference + float(values['-DEFL-'])), 'DEFL')
    tps_error_log(TwTpsSetTargetValue(tps1rc['DEFLFL'], V_tofreference + V_reference + float(values['-DEFLFL-'])), 'DEFLFL')
    tps_error_log(TwTpsSetTargetValue(tps1rc['RG'], V_rg), 'RG')
    if (polarity == 1):  # TPS does not switch LVS pulser channels in neg mode
        tps_error_log(TwTpsSetTargetValue(tps1rc['TOFREF'], V_tofreference), 'TOFREF')
        tps_error_log(TwTpsSetTargetValue(tps1rc['TOFEXTR1'], V_tofextractor1), 'TOFEXTR1')
    else:
        tps_error_log(TwTpsSetTargetValue(tps1rc['TOFREF'], V_tofextractor1), 'TOFREF')
        tps_error_log(TwTpsSetTargetValue(tps1rc['TOFEXTR1'], V_tofreference), 'TOFEXTR1')
    if (devmode):
        # Actual TPS voltages
        tps_debug = {'ORIFICE':float(values['-ORIFICE-']), 'IONEX':V_extractor + float(values['-IONEX-']), 
            'L1':V_lens1, 'INNER_CYL':V_extractor + V1, 'OUTER_CYL':V_extractor + V2,
            'MATSUDA':V_matsuda, 'REFERENCE':V_tofreference + V_reference,
            'DEFL':V_tofreference + V_reference + float(values['-DEFL-']),
            'DEFLFL':V_tofreference + V_reference + float(values['-DEFLFL-']),
            'RG': V_rg, 'RB': float(values['-RB-']),
            'DEFL1U': V_extractor + float(values['-DEFL1U-']), 'DEFL1D': V_extractor + float(values['-DEFL1D-']),
            'DEFL1R': V_extractor + float(values['-DEFL1R-']), 'DEFL1L': V_extractor + float(values['-DEFL1L-']),
            'L2': V_tofreference + V_reference + float(values['-LENS2-']),
            'MCP':float(values['-MCP-']), 'PA':float(values['-PA-']), 'DRIFT': float(values['-DRIFT-']),
            'TOFEXTR2': float(values['-TOFEXTR2-']), 'TOFPULSE': float(values['-TOFPULSE-']),
            'POLARITY': get_ionmode(), 'HVPOS': float(values['-HVPOS-']), 'HVNEG': float(values['-HVNEG-'])}
        if (polarity == 1):
            tps_debug['TOFREF'] = V_tofreference
            tps_debug['TOFEXTR1'] = V_tofextractor1
        else:
            tps_debug['TOFREF'] = V_tofextractor1
            tps_debug['TOFEXTR1'] = V_tofreference
        with open('TPS2_debug.txt', 'w') as f:
            json.dump(tps_debug, f, indent = 2)
        for key,value in tps_debug.items():
            log.debug(f"TPS_{key}: {value}")

def set_voltages_tof(values):
    """
    Set all ion_energy-independent (tof) voltages.
    """
    tps_error_log(TwTpsSetTargetValue(tps1rc['RB'], float(values['-RB-'])), 'RB')
    tps_error_log(TwTpsSetTargetValue(tps1rc['TOFPULSE'], float(values['-TOFPULSE-'])), 'TOFPULSE')
    tps_error_log(TwTpsSetTargetValue(tps1rc['TOFEXTR2'], float(values['-TOFEXTR2-'])), 'TOFEXTR2')
    tps_error_log(TwTpsSetTargetValue(tps1rc['DRIFT'], float(values['-DRIFT-'])), 'DRIFT')
    tps_error_log(TwTpsSetTargetValue(tps1rc['PA'], float(values['-PA-'])), 'PA')
    tps_error_log(TwTpsSetTargetValue(tps1rc['MCP'], float(values['-MCP-'])), 'MCP')
    tps_error_log(TwTpsSetTargetValue(tps1rc['HVSUPPLY'], float(values['-HVSUPPLY-'])), 'HVSUPPLY')
    tps_error_log(TwTpsSetTargetValue(tps1rc['HVPOS'], float(values['-HVPOS-'])), 'HVPOS')
    tps_error_log(TwTpsSetTargetValue(tps1rc['HVNEG'], float(values['-HVNEG-'])), 'HVNEG')

def load_setpoints(set_file):
    """Load setpoints from file"""
    with open(set_file, 'r') as f:
        setpoints = json.load(f)
    return setpoints

def save_setpoints(set_file, setpoints, values):
    """Save setpoints to file"""
    for key in SETPOINTS:
        setpoints[key] = values[key]
    with open(set_file, 'w') as f:
        json.dump(setpoints, f, indent = 2)

def read_setpoints_from_tps():
    """Read current setpoints from TPS"""
    if (devmode):
        tps2setpoint = copy.deepcopy(tps1rc)
        with open('TPS2_debug.txt', 'r') as f:
            tps2setpoint = json.load(f)
    else:
        tps2setpoint = copy.deepcopy(tps1rc)
        value = np.zeros(1, dtype=np.float64)
        for key in tps1rc:
            rv = TwTpsGetTargetValue(tps1rc[key], value)
            if (rv != TwSuccess): 
                log.error(f'Failed to read {key} voltage: {TwTranslateReturnValue(rv).decode()}.')
            tps2setpoint[key] = value[0]
        tps2setpoint['POLARITY'] = get_ionmode()
    return tps2setpoint

def zero_all():
    """Zero all voltages"""
    for key in tps1rc:
        rv = TwTpsSetTargetValue(tps1rc[key], 0)
        tps_error_log(rv, key)

def make_window():
    """Make GUI window"""
    menu_def = [['&Help', ['&Help', '&About']]]
    layout = [[sg.Menu(menu_def, key='-MENU-')]]
    layout += [[sg.Frame('Energy per charge (eV/e)', 
        [[sg.Text('Ion energy', size=(12,1)), sg.Input(settings.get('-ION_ENERGY-', '20'), size=(8,1), key='-ION_ENERGY-'),
        sg.Text('ESA energy', size=(15,1)), sg.Input(settings.get('-ESA_ENERGY-', '100'), size=(8,1), key='-ESA_ENERGY-'), 
        sg.Text('TOF energy', size=(15,1)), sg.Input(settings.get('-TOF_ENERGY-', '55'), size=(8,1), key='-TOF_ENERGY-')]], expand_x=True
        )]]
    layout += [[sg.Frame('Voltages (V)',
        [[sg.Text('Orifice', size=(12,1)), sg.Input(settings.get('-ORIFICE-', '0'), size=(8,1), key='-ORIFICE-'),
        sg.Text('Matsuda', size=(15,1)), sg.Input(settings.get('-MATSUDA-', '0'), size=(8,1), key='-MATSUDA-'),
        sg.Text('TOF Pulse', size=(15,1)), sg.Input(settings.get('-TOFPULSE-', '0'), size=(8,1), key='-TOFPULSE-')],
        [sg.Text('Lens 1', size=(12,1)), sg.Input(settings.get('-LENS1-', '0'), size=(8,1), key='-LENS1-'),
        sg.Text('Lens 2', size=(15,1)), sg.Input(settings.get('-LENS2-', '0'), size=(8,1), key='-LENS2-'),
        sg.Text('RG', size=(15,1)), sg.Input(settings.get('-RG-', '0'), size=(8,1), key='-RG-')],
        [sg.Text('Deflector 1 up', size=(12,1)), sg.Input(settings.get('-DEFL1U-', '0'), size=(8,1), key='-DEFL1U-'),
        sg.Text('Deflector 2', size=(15,1)), sg.Input(settings.get('-DEFL-', '0'), size=(8,1), key='-DEFL-'),
        sg.Text('RB', size=(15,1)), sg.Input(settings.get('-RB-', '0'), size=(8,1), key='-RB-')],
        [sg.Text('Deflector 1 down', size=(12,1)), sg.Input(settings.get('-DEFL1D-', '0'), size=(8,1), key='-DEFL1D-'),
        sg.Text('Deflector Flange 2', size=(15,1)), sg.Input(settings.get('-DEFLFL-', '0'), size=(8,1), key='-DEFLFL-'),
        sg.Text('Drift', size=(15,1)), sg.Input(settings.get('-DRIFT-', '0'), size=(8,1), key='-DRIFT-')],
        [sg.Text('Deflector 1 left', size=(12,1)), sg.Input(settings.get('-DEFL1L-', '0'), size=(8,1), key='-DEFL1L-'),
        sg.Text('Reference', size=(15,1)), sg.Input(settings.get('-REF-', '0'), size=(8,1), key='-REF-'),
        sg.Text('PA', size=(15,1)), sg.Input(settings.get('-PA-', '0'), size=(8,1), key='-PA-')],
        [sg.Text('Deflector 1 right', size=(12,1)), sg.Input(settings.get('-DEFL1R-', '0'), size=(8,1), key='-DEFL1R-'),
        sg.Text('TOF Extractor 1', size=(15,1)), sg.Input(settings.get('-TOFEXTR1-', '0'), size=(8,1), key='-TOFEXTR1-'),
        sg.Text('MCP', size=(15,1)), sg.Input(settings.get('-MCP-', '0'), size=(8,1), key='-MCP-')],
        [sg.Text('Ion Extractor', size=(12,1)), sg.Input(settings.get('-IONEX-', '0'), size=(8,1), key='-IONEX-'), 
        sg.Text('TOF Extractor 2', size=(15,1)), sg.Input(settings.get('-TOFEXTR2-', '0'), size=(8,1), key='-TOFEXTR2-')],
        [sg.Input(settings.get('-HVPOS-', '1000'), visible=False, key='-HVPOS-'),
        sg.Input(settings.get('-HVNEG-', '-4000'), visible=False, key='-HVNEG-'),
        sg.Input(settings.get('-HVSUPPLY-', '1'), visible=False, key='-HVSUPPLY-')]], expand_x=True
        )]]
    layout += [[sg.Button('Send all', key='-SET_TPS-'), sg.Button('Read setpoints', key='-READ_FROM_TPS-'),  
        sg.Input(visible=False, enable_events=True, key='-LOAD-'), sg.FileBrowse('Open...', initial_folder='setpoints', target='-LOAD-', key = '-LOAD2-'), 
        sg.Input(visible=False, enable_events=True, key='-SAVE-'), sg.FileSaveAs('Save...', default_extension = 'txt', initial_folder='setpoints', target='-SAVE-', key = '-SAVE2-'),
        sg.Button('Zero all', key='-ZERO_ALL-')]]
    layout += [[sg.Frame('Scan', 
        [[sg.Text('Scanning steps', size=(12,1), justification='left'), sg.Input(enable_events=True, default_text=settings.get('-SCAN_FILE-', 'steps.txt'), 
            tooltip='Text file with 1st column = ion energy (eV) and 2nd column = step duration (s)', justification='left', key='-SCAN_FILE-', expand_x=True), 
        sg.FileBrowse('Browse...', initial_folder='steps', target='-SCAN_FILE-', key = '-SCAN_FILE2-'), sg.Button('Edit', key='-EDIT-')],
        [sg.Text('Datafile name', size=(12,1), justification='left'), sg.Input(settings.get('-DATAFILE_NAME-', 'EMSscan_<year>-<month>-<day>_<hour>h<minute>m<second>s.h5'), 
            expand_x=True, justification='left', key='-DATAFILE_NAME-')],
        [sg.Button('Start', size=(4,1), key='-START-'), sg.Button('Cancel', key='-STOP-'),
        sg.ProgressBar(max_value=100, orientation='h', size=(20, 10), key='-PROGRESS_BAR-', expand_x=True, bar_color=('#FAC761', '#FFFFFF'))]], expand_x=True
        )]]
    layout += [[sg.Multiline(size=(50,5), autoscroll=True, 
        reroute_stdout=True, echo_stdout_stderr=True, write_only=True, key='-LOG_OUTPUT-',
        right_click_menu=['', ['&Clear']], background_color=sg.theme_background_color(), 
        text_color=sg.theme_element_text_color(), no_scrollbar=True, expand_x=True, expand_y=True)]]

    return sg.Window('EMS scan | TOFWERK', layout, icon=resource_path('tw.ico'), resizable=True, finalize=True, 
        return_keyboard_events=False, enable_close_attempted_event=True, text_justification='right')

def bind_mouse_wheel(window):
    """Bind mouse wheel to text inputs"""
    for key in SETPOINTS:
        window[key].bind('<MouseWheel>', ',+MOUSE WHEEL+')

def scanning_thread(window, values):
    """Energy scanning"""

    if (not TwTofDaqRunning() and not devmode):
        log.error('TofDaqRec not running.')
        return

    polarity = get_ionmode()
    if (polarity==0):
        return

    # read scanning script file. 1st column: ion energy (eV), 2nd column: step duration (s)
    try:  
        scandf = pd.read_table(values['-SCAN_FILE-'], delim_whitespace = True, names = ['energy', 'time'])
    except:
        log.error(f"Error reading scan script: {values['-SCAN_FILE-']}")
        return
    # some sanity checks
    if (not pd.to_numeric(scandf.energy, errors='coerce').notnull().all() or 
        not pd.to_numeric(scandf.time, errors='coerce').notnull().all() or
        (scandf.time<=0).any()):
        log.error(f"Error reading scan script: {values['-SCAN_FILE-']}")
        return
    energy_steps = scandf.energy
    step_duration = scandf.time
    time_remaining = step_duration.sum()
        
    for key, state in {'-START-': True, '-STOP-': False}.items(): window[key].update(disabled=state)
    progress = 0
    log.info('Energy scan started.')

    TwTpsSaveSetFile('TwTpsTempSetFile'.encode())
    setpoints = copy.deepcopy(SETPOINTS)
    save_setpoints('./TempScanSetFile'.encode(), setpoints, values)

    set_voltages_ea(values, energy_steps[0], polarity)
    set_voltages_tof(values)
    for key in SETPOINTS: window[key].update(background_color='#99C794')
    window['-ION_ENERGY-'].update(value=energy_steps[0])

    # start acquisition
    if TwDaqActive():
        log.warning('Stopping already running acquisition...')
        TwStopAcquisition()
        while TwDaqActive():  # wait until acquisition is stopped
            if exit_event.wait(timeout=1): break

    TwSaveIniFile(''.encode())
    TwSetDaqParameter('DataFileName'.encode(), values['-DATAFILE_NAME-'].encode())

    # Check if EnergyData is a registered data source
    nbrElements = np.zeros((1,), dtype=np.int32)
    rv = TwQueryRegUserDataSize('/EnergyData'.encode(), nbrElements)
    if (rv != TwSuccess and not devmode):
        rv = TwRegisterUserDataBufPy('/EnergyData', ['Ion energy (eV)', 'ESA energy (eV)'], 0)
        if (rv != TwSuccess and not devmode):
            log.error(f"Failed to register data source '/EnergyData': {TwTranslateReturnValue(rv).decode()}.")
        
    exit_event.wait(timeout=2)  # initial delay, to make sure all voltages are set.

    TwStartAcquisition()
    log.info('Starting TofDaq acquisition.')
    while (not TwDaqActive() and not devmode):  # wait until acquisition is started
        if exit_event.wait(timeout=1): break

    TwAddAttributeString('/EnergyData'.encode(), 'ScanningScript'.encode(), values['-SCAN_FILE-'].encode())

    exit_event.wait(timeout=1)  # additional delay, to make sure TofDaq is ready.
    # start energy scan
    window['-ION_ENERGY-'].update(background_color='#FAC761')  # orange
    for idx, step in enumerate(energy_steps):
        log.info(f'Scanning step {step:.1f} eV    {progress:.0f} %    {time_remaining:.0f} s remaining.')
        h5logtext = f'{step:g} eV'.encode()
        TwAddLogEntry(h5logtext, 0)
        set_voltages_ea(values, step, polarity)
        window['-ION_ENERGY-'].update(value=round(step, 2))
        TwUpdateUserData('/EnergyData'.encode(), 2, np.array([step, values['-ESA_ENERGY-']], dtype=np.float64))
        window['-PROGRESS_BAR-'].update_bar(progress, 100)
        time_remaining -= step_duration[idx]
        progress = 100*(1-time_remaining/step_duration.sum())
        if exit_event.wait(timeout=step_duration[idx]): break
    window['-PROGRESS_BAR-'].update_bar(100, 100)       
    log.info('Stopping acquisition.')
    window['-ION_ENERGY-'].update(background_color='#99C794')  # back to green
    desc = TSharedMemoryDesc()
    TwGetDescriptor(desc)
    TwStopAcquisition()
    while (TwDaqActive() and not devmode):  # wait until acquisition is stopped
        time.sleep(1)
    TwLoadIniFile(''.encode())
    TwTpsLoadSetFile('TwTpsTempSetFile'.encode())
    if os.path.exists('./TwTpsTempSetFile'): os.remove('./TwTpsTempSetFile')
    setpoints = load_setpoints('./TempScanSetFile'.encode())
    if os.path.exists('./TempScanSetFile'): os.remove('./TempScanSetFile')
    for key in SETPOINTS:
        window[key].update(value=setpoints[key])
    TwUpdateUserData('/EnergyData'.encode(), 2, np.array([values['-ION_ENERGY-'], values['-ESA_ENERGY-']], dtype=np.float64))
    log.info('Energy scan completed.')
    log.info(f'Datafile: {desc.currentDataFileName.decode()}')
    settings['-LASTDATAFILE-'] = desc.currentDataFileName.decode()
    [window[key].update(disabled=value) for key, value in {'-START-': False, '-STOP-': True}.items()]
    window['-PROGRESS_BAR-'].update_bar(0, 100)
    exit_event.clear()  # clear exit flag

def main():
    window = make_window()

    setpoints = SETPOINTS

    logging.basicConfig(stream=sys.stdout, format='%(asctime)s [%(levelname)s] %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    ods = DebugViewHandler()
    log.addHandler(ods)

    # connect to TPS2
    tps_ip = 'localhost'  # TPS2 host name or IP
    if TwTofDaqRunning():
        rv = TwTpsConnect2(tps_ip.encode(), 1)
        if rv != TwSuccess:
            log.error('Failed to connect to TPS2.')
        else:
            log.info(f'TPS2 connected via {tps_ip}.')
            # TwTpsSaveSetFile('TwTpsTempSetFile'.encode())
    else:
        if devmode:
            log.warning('Developer mode.')
        else:
            log.error('TofDaqRec not running.')

    for key, state in {'-START-': False, '-STOP-': True}.items():
        window[key].update(disabled=state)

    # keyboard shortcuts
    window.bind('<Control-o>', '+OPEN+')  
    window.bind('<Control-s>', '+SAVE+')
    window.bind('<Control-z>', '+ZERO+')
    window.bind('<Control-r>', '+READ+')
    window.bind('<Return>',    '+SEND+')
    window.bind('<Button-2>',  '+SEND2+')  # mouse scroll wheel

    # bind mouse wheel to element keys
    bind_mouse_wheel(window)

    # Store Ion and ESA energy as a registered data source
    rv = TwRegisterUserDataBufPy('/EnergyData', ['Ion energy (eV)', 'ESA energy (eV)'], 0)
    if (rv != TwSuccess and not devmode): 
        log.error(f"Failed to register data source '/EnergyData': {TwTranslateReturnValue(rv).decode()}.")

    # Event Loop 
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSE_ATTEMPTED_EVENT:
            break
        elif event == 'About':
            sg.popup_no_buttons('EMS scan software', 'Version ' + __version__, 'Author: ' + __author__,
                __copyright__, title = 'About', icon = resource_path('tw.ico'), image = resource_path('tw.png'), non_blocking = True)
        elif event == 'Help':
            f = open(resource_path('help.txt'), 'r')
            sg.popup_no_buttons(f.read(), title = 'Help', icon = resource_path('tw.ico'), 
                line_width = 80, font = ('Courier', 10), non_blocking = True)
            f.close()
        elif event == '-START-':
            if is_all_numeric(values):
                threading.Thread(target=scanning_thread, args=(window,values,), daemon=True).start()
        elif event == '-STOP-':
            exit_event.set()
            log.warning('Stopping energy scan by user request.')
            window['-ION_ENERGY-'].update(background_color='#FFFFFF')
        elif event == 'Clear':
            window['-LOG_OUTPUT-'].update('')
        elif event == '+SAVE+':  # Ctrl-s
            window['-SAVE2-'].Click()  # generate -SAVE- event
        elif event == '-SAVE-':
            if values[event]!='':
                save_setpoints(values[event], setpoints, values)
                log.info(f'Set values saved to {os.path.basename(values[event])}')
            window['-SAVE-'].update('')  # fix for cancel button issue (https://github.com/PySimpleGUI/PySimpleGUI/issues/3366)
        elif event == '+OPEN+':  # Ctrl-o
            window['-LOAD2-'].Click()  # generate -LOAD- event
        elif event == '-LOAD-':
            if values[event]!='':
                setpoints=load_setpoints(values[event])
                for key in SETPOINTS:
                    window[key].update(value=setpoints[key])
                log.info(f'Set values loaded from {os.path.basename(values[event])}')
            window['-LOAD-'].update('')
        elif event in ('-SET_TPS-', '+SEND+', '+SEND2+'):
            if is_all_numeric(values):
                polarity = get_ionmode()
                if (polarity!=0):
                    ion_energy = float(values['-ION_ENERGY-'])
                    set_voltages_ea(values, ion_energy, polarity)
                    set_voltages_tof(values)
                    log.info('TPS voltages set.')
                    for key in SETPOINTS:
                        window[key].update(background_color='#99C794')
                    TwUpdateUserData('/EnergyData'.encode(), 2, np.array([values['-ION_ENERGY-'], values['-ESA_ENERGY-']], dtype=np.float64))
        elif event in ('-READ_FROM_TPS-', '+READ+'):  # Ctrl-r
            tps2setpoint = read_setpoints_from_tps()
            if (tps2setpoint['POLARITY']!=0):
                polarity = 1 if (tps2setpoint['POLARITY']==1) else -1
                rg_correction = 0.25*polarity  # ion energy correction of RG in V/eV
                esa_energy, ion_energy = calculate_energies_from_ESA_voltages(tps2setpoint['INNER_CYL'], tps2setpoint['OUTER_CYL'], polarity)
                if (polarity == 1):  # TOF Reference and TOF Extr. 1 are switched in neg mode.
                    tofref = tps2setpoint['TOFREF']
                    tofextr1 = tps2setpoint['TOFEXTR1']
                else:
                    tofref = tps2setpoint['TOFEXTR1']
                    tofextr1 = tps2setpoint['TOFREF']
                tof_energy = ion_energy - tofref*polarity
                V_extractor = (ion_energy - esa_energy)*polarity
                energy_offset = calculate_energy_offset(esa_energy)
                window['-ESA_ENERGY-'].update(value=round(esa_energy, 2))
                window['-ION_ENERGY-'].update(value=round(ion_energy - energy_offset, 2))
                window['-MCP-'].update(value=round(tps2setpoint['MCP'], 3))
                window['-PA-'].update(value=round(tps2setpoint['PA'], 3))
                window['-DRIFT-'].update(value=round(tps2setpoint['DRIFT'], 3))
                window['-TOFEXTR2-'].update(value=round(tps2setpoint['TOFEXTR2'], 3))
                window['-TOFPULSE-'].update(value=round(tps2setpoint['TOFPULSE'], 3))
                window['-RB-'].update(value=round(tps2setpoint['RB'], 3))
                window['-RG-'].update(value=round(tps2setpoint['RG'] - tofref*rg_correction, 3))
                window['-ORIFICE-'].update(value=round(tps2setpoint['ORIFICE'], 3))
                window['-LENS1-'].update(value=round(tps2setpoint['L1'] - V_extractor - 0.955*(esa_energy - 100)*polarity, 2))
                window['-DEFL1U-'].update(value=round(tps2setpoint['DEFL1U'] - V_extractor, 3))
                window['-DEFL1D-'].update(value=round(tps2setpoint['DEFL1D'] - V_extractor, 3))
                window['-DEFL1R-'].update(value=round(tps2setpoint['DEFL1R'] - V_extractor, 3))
                window['-DEFL1L-'].update(value=round(tps2setpoint['DEFL1L'] - V_extractor, 3))
                window['-MATSUDA-'].update(value=round(tps2setpoint['MATSUDA'] - V_extractor - 0.24*(esa_energy - 100)*polarity, 2))
                window['-LENS2-'].update(value=round(tps2setpoint['L2'] - tps2setpoint['REFERENCE'], 3))
                window['-DEFL-'].update(value=round(tps2setpoint['DEFL'] - tps2setpoint['REFERENCE'], 3))
                window['-DEFLFL-'].update(value=round(tps2setpoint['DEFLFL'] - tps2setpoint['REFERENCE'], 3))
                window['-REF-'].update(value=round(tps2setpoint['REFERENCE'] + (tof_energy - ion_energy)*polarity, 3))
                window['-TOF_ENERGY-'].update(value=round(tof_energy, 2))
                window['-TOFEXTR1-'].update(value=round(tofextr1 + (tof_energy - ion_energy)*polarity, 3))
                window['-IONEX-'].update(value=round(tps2setpoint['IONEX'] - V_extractor, 3))
                window['-HVPOS-'].update(value=round(tps2setpoint['HVPOS'], 3))
                window['-HVNEG-'].update(value=round(tps2setpoint['HVNEG'], 3))
                log.info('Updated set values from current TPS setpoints.')
        elif event in ('-ZERO_ALL-', '+ZERO+'):  # Ctrl-z
            log.info('All voltages set to zero.')
            zero_all()
            for key in SETPOINTS:
                window[key].update(background_color='#6699CC')
        # elif re.search('\+MOUSE WHEEL\+$', event) is not None:
        elif event.endswith('+MOUSE WHEEL+'):
            key = re.split(',', event)[0]
            if key == '-ION_ENERGY-':
                scroll_stepsize = 0.1
            else:
                scroll_stepsize = 1
            window[key].update(value=round(float(values[key]) - float(window[key].user_bind_event.delta/120*scroll_stepsize), 2))
        elif event == '-EDIT-':
            sg.execute_editor(values['-SCAN_FILE-'])

    # save user settings
    for key in SETPOINTS:
        settings[key] = values[key]
    settings['-DATAFILE_NAME-'] = values['-DATAFILE_NAME-']
    settings['-SCAN_FILE-'] = values['-SCAN_FILE-']
    TwUnregisterUserData('/EnergyData'.encode())
    TwTpsDisconnect()
    TwCleanupDll()

    window.close()
    sys.exit(0)

if __name__ == '__main__':
    main()
