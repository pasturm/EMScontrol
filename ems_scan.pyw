#! /usr/bin/env python3

"""EMS voltage control and energy scanning"""

__version__ = '0.6.8'
__author__ = 'Patrick Sturm'
__copyright__ = 'Copyright 2021-2022, TOFWERK'

import numpy as np
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


# Windows element keys that can change background color
V_INPUTS = {'-ORIFICE-':0, '-LENS1-':0, '-DEFL1U-':0, '-DEFL1D-':0, '-DEFL1L-':0, '-DEFL1R-':0, 
    '-MATSUDA-':0, '-LENS2-':0, '-DEFL-':0, '-DEFLFL-':0, '-REF-':0, '-TOFEXTR1-':0, '-RG-':0,
    '-RB-':0, '-TOFEXTR2-':0, '-TOFPULSE-':0, '-DRIFT-':0, '-PA-':0, '-MCP-':0, '-IONEX-':0,
    '-ESA_ENERGY-':0, '-TOF_ENERGY-':0, '-ION_ENERGY-':0}


# Window element keys that will be saved to a file
SETPOINTS = {**V_INPUTS, '-START_ENERGY-':0, '-END_ENERGY-':0, '-STEP_SIZE-':0, 
    '-TIME_PER_STEP-':0, '-HVSUPPLY-':0, '-HVPOS-':0, '-HVNEG-':0}


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


def calculate_ESA_voltages(esa_energy, r0 = 0.100, d = 0.0125):
    """
    Calculates the cylinder electrode potentials for a given energy.
    
    Arguments:
    ke: mean kinetic energy of singly charged ion, eV
    r0: mean cylinder radius, m
    d: half-distance between inner and outer cylinder, m
    
    Return values:
    V1, V2: inner and outer cylinder voltage, V

    from Yavor, Optics of Charged Particle Analyzers.
    """
    r1 = r0-d  # inner cylinder radius, m
    r2 = r0+d  # outer cylinder radius, m
    V1 = esa_energy*2*math.log(r2/r1)*(math.log(r1)-math.log(r0))/(math.log(r2)-math.log(r1))  # inner cylinder voltage, V
    V2 = esa_energy*2*math.log(r2/r1) + V1  # outer cylinder voltage, V
    return V1, V2


def calculate_energies_from_ESA_voltages(V_inner, V_outer):
    """
    Calculates the ESA energy and staring ion energy from the cylinder electrode potentials.
    
    Arguments:
    V_inner: inner cylinder potential, V
    V_outer: outer cylinder potential, V
    
    Return values:
    esa_energy, ion_energy : ESA and staring ion energy, eV
    """
    r0 = 0.100
    d = 0.0125
    r1 = r0-d  # inner cylinder radius, m
    r2 = r0+d  # outer cylinder radius, m
    c1 = 2*math.log(r2/r1)
    c2 = c1*(math.log(r1)-math.log(r0))/(math.log(r2)-math.log(r1))
    esa_energy = (V_outer-V_inner)/c1
    ion_energy = (V_inner*(1+(2*c2+c1-2)/c1)+V_outer*(1-(2*c2+c1-2)/c1))/2
    return esa_energy, ion_energy


def tps_error_log(rv, key):
    if (rv != TwSuccess and not devmode): log.error(f'Failed to set {key} voltage: {TwTranslateReturnValue(rv).decode()}.')


def set_voltages_ea(values, ion_energy):
    """
    Set all ion_energy-dependent voltages.
    """ 
    V1, V2 = calculate_ESA_voltages(float(values['-ESA_ENERGY-']))
    V_extractor = ion_energy - float(values['-ESA_ENERGY-'])
    V_reference = float(values['-REF-'])
    V_tofreference = ion_energy - float(values['-TOF_ENERGY-'])  # from LV channel -> with sign
    V_tofextractor1 = V_tofreference + float(values['-TOFEXTR1-'])  # from LV channel -> with sign, relative to TOF reference
    rg_correction = 0.25  # ion energy correction of RG in V/eV
    V_rg = float(values['-RG-']) + V_tofreference*rg_correction  # -RG- is set value if TOFREF = 0 V (ion energy - tof energy = 0 eV)

    tps_error_log(TwTpsSetTargetValue(tps1rc['ORIFICE'], float(values['-ORIFICE-'])), 'ORIFICE')
    tps_error_log(TwTpsSetTargetValue(tps1rc['IONEX'], V_extractor + float(values['-IONEX-'])), 'IONEX')
    tps_error_log(TwTpsSetTargetValue(tps1rc['L1'], V_extractor + float(values['-LENS1-']) + 0.955*(float(values['-ESA_ENERGY-'])-100)), 'L1')
    tps_error_log(TwTpsSetTargetValue(tps1rc['DEFL1U'], V_extractor + float(values['-DEFL1U-'])), 'DEFL1U')
    tps_error_log(TwTpsSetTargetValue(tps1rc['DEFL1D'], V_extractor + float(values['-DEFL1D-'])), 'DEFL1D')
    tps_error_log(TwTpsSetTargetValue(tps1rc['DEFL1R'], V_extractor + float(values['-DEFL1R-'])), 'DEFL1R')
    tps_error_log(TwTpsSetTargetValue(tps1rc['DEFL1L'], V_extractor + float(values['-DEFL1L-'])), 'DEFL1L')
    tps_error_log(TwTpsSetTargetValue(tps1rc['INNER_CYL'], V_extractor + V1), 'INNER_CYL')
    tps_error_log(TwTpsSetTargetValue(tps1rc['OUTER_CYL'], V_extractor + V2), 'OUTER_CYL')
    tps_error_log(TwTpsSetTargetValue(tps1rc['MATSUDA'], V_extractor + float(values['-MATSUDA-']) + 0.24*(float(values['-ESA_ENERGY-'])-100)), 'MATSUDA')
    tps_error_log(TwTpsSetTargetValue(tps1rc['REFERENCE'], V_tofreference + V_reference), 'REFERENCE')
    tps_error_log(TwTpsSetTargetValue(tps1rc['L2'], V_tofreference + V_reference + float(values['-LENS2-'])), 'L2')
    tps_error_log(TwTpsSetTargetValue(tps1rc['DEFL'], V_tofreference + V_reference + float(values['-DEFL-'])), 'DEFL')
    tps_error_log(TwTpsSetTargetValue(tps1rc['DEFLFL'], V_tofreference + V_reference + float(values['-DEFLFL-'])), 'DEFLFL')
    tps_error_log(TwTpsSetTargetValue(tps1rc['TOFREF'], V_tofreference), 'TOFREF')
    tps_error_log(TwTpsSetTargetValue(tps1rc['TOFEXTR1'], V_tofextractor1), 'TOFEXTR1')
    tps_error_log(TwTpsSetTargetValue(tps1rc['RG'], V_rg), 'RG')
    # # Show actual TPS voltages as debug message: Orifice|Extractor|Lens1|Inner|Outer|Matsuda|Reference|Lens2|Defl|Deflfl|TOFreference|TOFExtr1|RG
    # log.debug(f"\nOrifice:{values['-ORIFICE-']}\nExtractor:{V_extractor + float(values['-IONEX-']):g}"
    #     f"\nLens1:{V_extractor + float(values['-LENS1-']) + 0.9*(float(values['-ESA_ENERGY-'])-100):g}"
    #     f"\nInner:{V_extractor + V1:g}\nOuter:{V_extractor + V2:g}"
    #     f"\nMatsuda:{V_extractor + float(values['-MATSUDA-']) + 0.25*(float(values['-ESA_ENERGY-'])-100):g}"
    #     f"\nReference:{V_tofreference + V_reference:g}\nLens2:{V_tofreference + V_reference + float(values['-LENS2-']):g}"
    #     f"\nDefl:{V_tofreference + V_reference + float(values['-DEFL-']):g}\nDeflfl:{V_tofreference + V_reference + float(values['-DEFLFL-']):g}"
    #     f"\nTOFref:{V_tofreference:g}\nTOFExtr1:{V_tofextractor1:g}\nRG:{V_rg:g}")


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
        json.dump(setpoints, f)


def read_setpoints_from_tps():
    """Read current setpoints from TPS"""
    tps2setpoint = copy.deepcopy(tps1rc)
    value = np.zeros(1, dtype=np.float64)
    for key in tps1rc:
        rv = TwTpsGetTargetValue(tps1rc[key], value)
        if (rv != TwSuccess and not devmode): 
            log.error(f'Failed to read {key} voltage: {TwTranslateReturnValue(rv).decode()}.')
        tps2setpoint[key] = value[0]
    return tps2setpoint


def zero_all():
    """Zero all voltages"""
    for key in tps1rc:
        rv = TwTpsSetTargetValue(tps1rc[key], 0)
        tps_error_log(rv, key)


def make_window():
    """Make GUI window"""
    sg.SetOptions(text_justification='right')
    # sg.theme('SystemDefaultForReal')

    menu_def = [['&Help', ['&Keyboard shortcuts...', '&Voltage mapping...', '&About...']]]
    layout = [[sg.Menu(menu_def, key='-MENU-')]]

    layout += [[sg.Frame('Energies (eV)', 
        [[sg.Text('Ion energy', size=(15,1)), sg.Input(default_text='50', size=(8,1), key='-ION_ENERGY-'),
        sg.Text('ESA energy', size=(15,1)), sg.Input(default_text='100', size=(8,1), key='-ESA_ENERGY-'), 
        sg.Text('TOF energy', size=(15,1)), sg.Input(default_text='60', size=(8,1), key='-TOF_ENERGY-')]]
        )]]

    layout += [[sg.Frame('Voltages (V)',
        [[sg.Text('Orifice', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-ORIFICE-'),
        sg.Text('Matsuda', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-MATSUDA-'),
        sg.Text('TOF Pulse', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-TOFPULSE-')],
        [sg.Text('Lens 1', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-LENS1-'),
        sg.Text('Lens 2', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-LENS2-'),
        sg.Text('RG', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-RG-')],
        [sg.Text('Deflector 1 up', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-DEFL1U-'),
        sg.Text('Deflector 2', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-DEFL-'),
        sg.Text('RB', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-RB-')],
        [sg.Text('Deflector 1 down', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-DEFL1D-'),
        sg.Text('Deflector Flange 2', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-DEFLFL-'),
        sg.Text('Drift', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-DRIFT-')],
        [sg.Text('Deflector 1 left', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-DEFL1L-'),
        sg.Text('Reference', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-REF-'),
        sg.Text('PA', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-PA-')],
        [sg.Text('Deflector 1 right', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-DEFL1R-'),
        sg.Text('TOF Extractor 1', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-TOFEXTR1-'),
        sg.Text('MCP', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-MCP-')],
        [sg.Text('Ion Extractor', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-IONEX-'), 
        sg.Text('TOF Extractor 2', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-TOFEXTR2-')],
        [sg.Input(visible=False, default_text='1000', key='-HVPOS-'),
        sg.Input(visible=False, default_text='-4000', key='-HVNEG-'),
        sg.Input(visible=False, default_text='1', key='-HVSUPPLY-')]]
        )]]
  
    layout += [[sg.Button('Send all', key='-SET_TPS-'), sg.Button('Read setpoints', key='-READ_FROM_TPS-'),  
        sg.Input(visible=False, enable_events=True, key='-LOAD-'), sg.FileBrowse('Open...', initial_folder='setpoints', target='-LOAD-', key = '-LOAD2-'), 
        sg.Input(visible=False, enable_events=True, key='-SAVE-'), sg.FileSaveAs('Save...', default_extension = 'tps', initial_folder='setpoints', target='-SAVE-', key = '-SAVE2-'),
        sg.Button('Zero all', key='-ZERO_ALL-')]]

    layout += [[sg.Frame('Scan', 
        [[sg.Text('Start ion energy (eV)', size=(15,1)), sg.Input(default_text='0', size=(8,1), key='-START_ENERGY-'),
        sg.Text('End ion energy (eV)', size=(15,1)), sg.Input(default_text='10', size=(8,1), key='-END_ENERGY-'),
        sg.Text('Step size (eV)', size=(15,1)), sg.Input(default_text='0.5', size=(8,1), key='-STEP_SIZE-')],
        [sg.Text('Time per step (s)', size=(15,1)), sg.Input(default_text='3', size=(8,1), key='-TIME_PER_STEP-')],
        [sg.Button('Start', key='-START-'), sg.Button('Cancel', key='-STOP-'),
        sg.ProgressBar(max_value=100, orientation='h', size=(20, 10), key='-PROGRESS_BAR-', expand_x=True, bar_color=('#FAC761', '#FFFFFF'))]]
        )]]

    layout += [[sg.Multiline(size=(50,5), autoscroll=True, 
        reroute_stdout=True, echo_stdout_stderr=True, write_only=True, key='-LOG_OUTPUT-',
        right_click_menu=['', ['&Clear']], background_color=sg.theme_background_color(), 
        text_color=sg.theme_element_text_color(), no_scrollbar=True, expand_x=True)]]

    return sg.Window('EMS scan | TOFWERK', layout, icon=resource_path('tw.ico'), resizable=True, finalize=True, 
        return_keyboard_events=False, enable_close_attempted_event=True)


def bind_mouse_wheel(window):
    """Bind mouse wheel to text inputs"""
    for key in SETPOINTS:
        window[key].bind('<MouseWheel>', ',+MOUSE WHEEL+')


def scanning_thread(window, values):
    """Energy scanning"""
    if (not TwTofDaqRunning() and not devmode):
        log.error('TofDaqRec not running.')
        return

    for key, state in {'-START-': True, '-STOP-': False}.items(): window[key].update(disabled=state)

    progress = 0
    start_energy = float(values['-START_ENERGY-'])  # start energy, eV
    end_energy = float(values['-END_ENERGY-'])  # end energy, eV
    step_size = float(values['-STEP_SIZE-'])  # energy step stize, eV
    time_per_step = float(values['-TIME_PER_STEP-'])  # time per energy step, s
    n_steps = math.floor((end_energy - start_energy)/step_size) + 1
    if step_size == 0:
        log.error('Step size must not be 0 eV.')
        [window[key].update(disabled=value) for key, value in {'-START-': False, '-STOP-': True}.items()]
        return
    if time_per_step <= 0:
        log.error('Time per step must be larger than 0 s.')
        [window[key].update(disabled=value) for key, value in {'-START-': False, '-STOP-': True}.items()]
        return
    log.info('Energy scan started.')

    TwTpsSaveSetFile('TwTpsTempSetFile'.encode())
    save_setpoints('./currentSetpoints.tps'.encode(), SETPOINTS, values)

    set_voltages_ea(values, start_energy)
    set_voltages_tof(values)
    for key in V_INPUTS: window[key].update(background_color='#99C794')
    window['-ION_ENERGY-'].update(value=values['-START_ENERGY-'])

    # start acquisition (-> one data file per scan) 
    if TwDaqActive():
        log.warning('Stopping already running acquisition...')
        TwStopAcquisition()
        while TwDaqActive():  # wait until acquisition is stopped
            if exit_event.wait(timeout=1): break

    TwSaveIniFile(''.encode())
    TwSetDaqParameter('DataFileName'.encode(), 'EMSscan_<year>-<month>-<day>_<hour>h<minute>m<second>s.h5'.encode())

    exit_event.wait(timeout=2)  # initial delay, to make sure all voltages are set.

    TwStartAcquisition()
    log.info('Starting TofDaq acquisition.')
    while (not TwDaqActive() and not devmode):  # wait until acquisition is started
        if exit_event.wait(timeout=1): break

    TwAddAttributeDouble('/EnergyData'.encode(), 'StartEnergy'.encode(), start_energy)
    TwAddAttributeDouble('/EnergyData'.encode(), 'EndEnergy'.encode(), end_energy)
    TwAddAttributeDouble('/EnergyData'.encode(), 'StepSize'.encode(), step_size)
    TwAddAttributeDouble('/EnergyData'.encode(), 'TimePerStep'.encode(), time_per_step)

    exit_event.wait(timeout=1)  # additional delay, to make sure TofDaq is ready.

    # start energy scan
    window['-ION_ENERGY-'].update(background_color='#FAC761')  # orange
    time_remaining = n_steps*time_per_step
    for i in step_size*np.arange(start_energy/step_size, end_energy/step_size+1e-6):
        log.info(f'Scanning step {i:.1f} eV    {progress:.0f} %    {time_remaining:.0f} s remaining.')
        h5logtext = f'{i:g} eV'.encode()
        TwAddLogEntry(h5logtext, 0)
        set_voltages_ea(values, i)
        window['-ION_ENERGY-'].update(value=round(i, 2))
        TwUpdateUserData('/EnergyData'.encode(), 2, np.array([i, values['-ESA_ENERGY-']], dtype=np.float64))
        window['-PROGRESS_BAR-'].update_bar(progress, 100)
        progress += 100 / n_steps
        time_remaining -= time_per_step
        if exit_event.wait(timeout=time_per_step): break
    window['-PROGRESS_BAR-'].update_bar(100, 100)       
    log.info('Stopping acquisition.')
    window['-ION_ENERGY-'].update(background_color='#99C794')  # back to green
    TwStopAcquisition()
    while (TwDaqActive() and not devmode):  # wait until acquisition is stopped
        if exit_event.wait(timeout=1): break
    TwLoadIniFile(''.encode())
    TwTpsLoadSetFile('TwTpsTempSetFile'.encode())
    if os.path.exists('./TwTpsTempSetFile'): os.remove('./TwTpsTempSetFile')
    setpoints = load_setpoints('./currentSetpoints.tps'.encode())
    for key in SETPOINTS:
        window[key].update(value=setpoints[key])
    TwUpdateUserData('/EnergyData'.encode(), 2, np.array([values['-ION_ENERGY-'], values['-ESA_ENERGY-']], dtype=np.float64))
    log.info('Energy scan completed.')
    [window[key].update(disabled=value) for key, value in {'-START-': False, '-STOP-': True}.items()]
    window['-PROGRESS_BAR-'].update_bar(0, 100)
    exit_event.clear()  # clear exit flag

def main():
    window = make_window()
    
    # load previous settings
    if os.path.exists('./currentSetpoints.tps'):
        setpoints = load_setpoints('./currentSetpoints.tps'.encode())
        for key in SETPOINTS:
            window[key].update(value=setpoints[key])
    else:
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
        elif event == 'About...':
            sg.popup_no_buttons('EMS scan software', 'Version ' + __version__,
                __copyright__, title = 'About', icon = resource_path('tw.ico'), image = resource_path('tw.png'), non_blocking = True)
        elif event == 'Keyboard shortcuts...':
            sg.popup_no_buttons(
                'Send all:       Enter or Scroll Wheel Click', 
                'Read setpoints: Ctrl-R',
                'Open...:        Ctrl-O',
                'Save...:        Ctrl-S', 
                'Zero all:       Ctrl-Z', 
                title = 'Keyboard shortcuts', icon = resource_path('tw.ico'), font = ('Courier', 10), non_blocking = True)
        elif event == 'Voltage mapping...':
            sg.popup_no_buttons(
                'TPS_Orifice            = Orifice',
                'TPS_Lens_1             = Lens_1 + V* + 0.955*(ESA_energy*V/eV - 100 V)',
                'TPS_Deflector_1_up     = Deflector_1_up + V*',
                'TPS_Deflector_1_down   = Deflector_1_down + V*',
                'TPS_Deflector_1_left   = Deflector_1_left + V*',
                'TPS_Deflector_1_right  = Deflector_1_right + V*',
                'TPS_Ion_Extractor      = Ion_Extractor + V*',
                'TPS_Matsuda            = Matsuda + V* + 0.24*(ESA_energy*V/eV - 100 V)',
                'TPS_Inner_Cylinder     = -0.26706*ESA_energy*V/eV + V*',
                'TPS_Outer_Cylinder     = 0.23558*ESA_energy*V/eV + V*',
                'TPS_TOF_Reference      = (Ion_energy - TOF_energy)*V/eV',
                'TPS_Reference          = Reference + TPS_TOF_Reference',
                'TPS_Lens_2             = Lens_2 + TPS_Reference',
                'TPS_Deflector_2        = Deflector_2 + TPS_Reference',
                'TPS_Deflector_Flange_2 = Deflector_Flange_2 + TPS_Reference',
                'TPS_TOF_Extractor_1    = TOF_Extractor_1 + TPS_TOF_Reference', 
                'TPS_TOF_Extractor_2    = TOF_Extractor_2',
                'TPS_TOF_Pulse          = TOF_Pulse',
                'TPS_RG                 = RG + TPS_TOF_Reference*0.25',
                'TPS_RB                 = RB',
                'TPS_Drift              = Drift',
                'TPS_PA                 = PA',
                'TPS_MCP                = MCP',
                'V* = (Ion_energy - ESA_energy)*V/eV',
                title = 'Voltage mapping', icon = resource_path('tw.ico'), 
                font = ('Courier', 10), non_blocking = True, line_width = 100)
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
                log.info('TPS voltages set.')
                ion_energy = float(values['-ION_ENERGY-'])
                set_voltages_ea(values, ion_energy)
                set_voltages_tof(values)
                for key in V_INPUTS:
                    window[key].update(background_color='#99C794')
                TwUpdateUserData('/EnergyData'.encode(), 2, np.array([values['-ION_ENERGY-'], values['-ESA_ENERGY-']], dtype=np.float64))
        elif event in ('-READ_FROM_TPS-', '+READ+'):  # Ctrl-r
            tps2setpoint = read_setpoints_from_tps()
            rg_correction = 0.25  # ion energy correction of RG in V/eV
            esa_energy, ion_energy = calculate_energies_from_ESA_voltages(tps2setpoint['INNER_CYL'], tps2setpoint['OUTER_CYL'])
            tof_energy = ion_energy - tps2setpoint['TOFREF']
            V_extractor = ion_energy - esa_energy
            window['-ESA_ENERGY-'].update(value=round(esa_energy, 2))
            window['-ION_ENERGY-'].update(value=round(ion_energy, 2))
            window['-MCP-'].update(value=round(tps2setpoint['MCP'], 3))
            window['-PA-'].update(value=round(tps2setpoint['PA'], 3))
            window['-DRIFT-'].update(value=round(tps2setpoint['DRIFT'], 3))
            window['-TOFEXTR2-'].update(value=round(tps2setpoint['TOFEXTR2'], 3))
            window['-TOFPULSE-'].update(value=round(tps2setpoint['TOFPULSE'], 3))
            window['-RB-'].update(value=round(tps2setpoint['RB'], 3))
            window['-RG-'].update(value=round(tps2setpoint['RG'] - tps2setpoint['TOFREF']*rg_correction, 3))
            window['-ORIFICE-'].update(value=round(tps2setpoint['ORIFICE'], 3))
            window['-LENS1-'].update(value=round(tps2setpoint['L1'] - V_extractor - 0.955*(esa_energy - 100), 2))
            window['-DEFL1U-'].update(value=round(tps2setpoint['DEFL1U'] - V_extractor, 3))
            window['-DEFL1D-'].update(value=round(tps2setpoint['DEFL1D'] - V_extractor, 3))
            window['-DEFL1R-'].update(value=round(tps2setpoint['DEFL1R'] - V_extractor, 3))
            window['-DEFL1L-'].update(value=round(tps2setpoint['DEFL1L'] - V_extractor, 3))
            window['-MATSUDA-'].update(value=round(tps2setpoint['MATSUDA'] - V_extractor - 0.24*(esa_energy - 100), 2))
            window['-LENS2-'].update(value=round(tps2setpoint['L2'] - tps2setpoint['REFERENCE'], 3))
            window['-DEFL-'].update(value=round(tps2setpoint['DEFL'] - tps2setpoint['REFERENCE'], 3))
            window['-DEFLFL-'].update(value=round(tps2setpoint['DEFLFL'] - tps2setpoint['REFERENCE'], 3))
            window['-REF-'].update(value=round(tps2setpoint['REFERENCE'] + tof_energy - ion_energy, 3))
            window['-TOF_ENERGY-'].update(value=round(tof_energy, 2))
            window['-TOFEXTR1-'].update(value=round(tps2setpoint['TOFEXTR1'] + tof_energy - ion_energy, 3))
            window['-IONEX-'].update(value=round(tps2setpoint['IONEX'] - V_extractor, 3))
            log.info('Updated set values from current TPS setpoints.')
        elif event in ('-ZERO_ALL-', '+ZERO+'):  # Ctrl-z
            log.info('All voltages set to zero.')
            zero_all()
            for key in V_INPUTS:
                window[key].update(background_color='#6699CC')
        # elif re.search('\+MOUSE WHEEL\+$', event) is not None:
        elif event.endswith('+MOUSE WHEEL+'):
            key = re.split(',', event)[0]
            if key in ('-ION_ENERGY-', '-STEP_SIZE-'):
                scroll_stepsize = 0.1
            else:
                scroll_stepsize = 1
            window[key].update(value=round(float(values[key]) - float(window[key].user_bind_event.delta/120*scroll_stepsize), 2))

    save_setpoints('./currentSetpoints.tps'.encode(), SETPOINTS, values)
    TwUnregisterUserData('/EnergyData'.encode())
    TwTpsDisconnect()
    TwCleanupDll()

    window.close()
    sys.exit(0)

if __name__ == '__main__':
    main()
