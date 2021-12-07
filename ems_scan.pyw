#! /usr/bin/python3-64

"""EMS voltage control and energy scanning"""

__version__ = '0.3.1'
__author__ = 'Patrick Sturm'
__copyright__ = 'Copyright 2021, TOFWERK'

import numpy as np
import time
import sys
import math
import logging
import threading
import os
import copy
import re
import PySimpleGUI as sg
from json import (load as jsonload, dump as jsondump)
from TofDaq import *
from TwTool import *


# Logger
log = logging.getLogger(__name__)


# TPS RC codes
tps1rc = {
'L2': 14,
'DEFL': 15,
'DEFLFL': 16,
'IONEX': 17,
'L1': 18,
'REFERENCE': 117,
'ORIFICE': 2500,
'INNER_CYL': 2501,
'OUTER_CYL': 2502,
'MATSUDA': 2503,
'DEFL1U': 2504,
'DEFL1D': 2505,
'DEFL1L': 2506,
'DEFL1R': 2507,
'TOFREF': 202,
'TOFEXTR1': 201,
'TOFEXTR2': 200,
'TOFPULSE': 203,
'RG': 2,
'RB': 1,
'DRIFT': 9,
'PA': 5,
'MCP': 6,
'HVSUPPLY': 602,
'HVPOS': 603,
'HVNEG': 604
}


# Windows element keys that are Voltages and can change background color
V_INPUTS = {'-ORIFICE-':0, '-LENS1-':0, '-DEFL1U-':0, '-DEFL1D-':0, '-DEFL1L-':0, '-DEFL1R-':0, 
    '-INNER_CYL-':0, '-OUTER_CYL-':0, '-MATSUDA-':0, '-LENS2-':0, '-DEFL-':0, '-DEFLFL-':0, '-REF-':0,
    '-TOFEXTR1-':0, '-RG-':0, '-RB-':0, '-TOFEXTR2-':0, '-TOFPULSE-':0, '-DRIFT-':0,
    '-PA-':0, '-MCP-':0, '-IONEX-':0}


# Window element keys that will be saved to a file
SETPOINTS = {'-ESA_ENERGY-':0, '-TOF_ENERGY-':0, '-ION_ENERGY-':0, '-POLARITY-':0, 
    '-ORIFICE-':0, '-LENS1-':0, '-DEFL1U-':0, '-DEFL1D-':0, '-DEFL1L-':0, '-DEFL1R-':0, 
    '-INNER_CYL-':0, '-OUTER_CYL-':0, '-MATSUDA-':0, '-LENS2-':0, '-DEFL-':0, '-DEFLFL-':0, '-REF-':0,
    '-START_ENERGY-':0, '-END_ENERGY-':0, '-STEP_SIZE-':0,'-TIME_PER_STEP-':0,
    '-TOFEXTR1-':0, '-RG-':0, '-RB-':0, '-TOFEXTR2-':0, '-TOFPULSE-':0, '-DRIFT-':0,
    '-PA-':0, '-MCP-':0, '-HVSUPPLY-':0, '-HVPOS-':0, '-HVNEG-':0, '-IONEX-':0}


# exit event to abort energy scanning
exit_event = threading.Event()


def calculate_EA_voltages(ea_energy, r0 = 0.100, d = 0.0125, polarity = 1):
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
    V1 = polarity*ea_energy*2*math.log(r2/r1)*(math.log(r1)-math.log(r0))/(math.log(r2)-math.log(r1))  # inner cylinder voltage, V
    V2 = polarity*ea_energy*2*math.log(r2/r1) + V1  # outer cylinder voltage, V
    return V1, V2


def tps_error_log(rv, key):
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc[key]}: {TwTranslateReturnValue(rv).decode()}.")


def set_voltages_ea(values, ion_energy):
    """
    Set all ion_energy-dependent voltages.
    """ 
    # polarity = 1 if (values['-POLARITY-']=='pos') else -1  # Note: does not yet work for neg
    V1, V2 = calculate_EA_voltages(float(values['-ESA_ENERGY-']), polarity=1)
    V_extractor = ion_energy - float(values['-ESA_ENERGY-'])
    V_reference = float(values['-REF-'])
    V_tofreference = ion_energy - float(values['-TOF_ENERGY-'])  # from LV channel -> with sign
    V_tofextractor1 = V_tofreference + float(values['-TOFEXTR1-'])  # from LV channel -> with sign, relative to TOF reference
    rg_correction = 0.25  # ion energy correction of RG in V/eV
    V_rg = float(values['-RG-']) + V_tofreference*rg_correction  # -RG- is set value if TOFREF = 0 V (ion energy - tof energy = 0 eV)

    rv = TwTpsSetTargetValue(tps1rc['ORIFICE'], float(values['-ORIFICE-']))
    tps_error_log(rv, 'ORIFICE')
    rv = TwTpsSetTargetValue(tps1rc['IONEX'], V_extractor + float(values['-IONEX-']))
    tps_error_log(rv, 'IONEX')
    rv = TwTpsSetTargetValue(tps1rc['L1'], V_extractor + float(values['-LENS1-']))
    tps_error_log(rv, 'L1')
    rv = TwTpsSetTargetValue(tps1rc['DEFL1U'], V_extractor + float(values['-DEFL1U-']))
    tps_error_log(rv, 'DEFL1U')
    rv = TwTpsSetTargetValue(tps1rc['DEFL1D'], V_extractor + float(values['-DEFL1D-']))
    tps_error_log(rv, 'DEFL1D')
    rv = TwTpsSetTargetValue(tps1rc['DEFL1R'], V_extractor + float(values['-DEFL1R-']))
    tps_error_log(rv, 'DEFL1R')
    rv = TwTpsSetTargetValue(tps1rc['DEFL1L'], V_extractor + float(values['-DEFL1L-']))
    tps_error_log(rv, 'DEFL1L')
    rv = TwTpsSetTargetValue(tps1rc['INNER_CYL'], V_extractor + V1 + float(values['-INNER_CYL-']))
    tps_error_log(rv, 'INNER_CYL')
    rv = TwTpsSetTargetValue(tps1rc['OUTER_CYL'], V_extractor + V2 + float(values['-OUTER_CYL-']))
    tps_error_log(rv, 'OUTER_CYL')
    rv = TwTpsSetTargetValue(tps1rc['MATSUDA'], V_extractor + float(values['-MATSUDA-']))
    tps_error_log(rv, 'MATSUDA')
    rv = TwTpsSetTargetValue(tps1rc['REFERENCE'], V_tofreference + V_reference)
    tps_error_log(rv, 'REFERENCE')
    rv = TwTpsSetTargetValue(tps1rc['L2'], V_tofreference + V_reference + float(values['-LENS2-']))
    tps_error_log(rv, 'L2')
    rv = TwTpsSetTargetValue(tps1rc['DEFL'], V_tofreference + V_reference + float(values['-DEFL-']))
    tps_error_log(rv, 'DEFL')
    rv = TwTpsSetTargetValue(tps1rc['DEFLFL'], V_tofreference + V_reference + float(values['-DEFLFL-']))
    tps_error_log(rv, 'DEFLFL')
    rv = TwTpsSetTargetValue(tps1rc['TOFREF'], V_tofreference)
    tps_error_log(rv, 'TOFREF')
    rv = TwTpsSetTargetValue(tps1rc['TOFEXTR1'], V_tofextractor1)
    tps_error_log(rv, 'TOFEXTR1')
    rv = TwTpsSetTargetValue(tps1rc['RG'], V_rg)
    tps_error_log(rv, 'RG')
    # Show actual TPS voltages as debug message: Orifice|Extractor|Lens1|Inner|Outer|Matsuda|Reference|Lens2|TOFreference|TOFExtr1|RG
    # log.debug(f"{values['-ORIFICE-']}|{V_extractor}|{V_extractor + float(values['-LENS1-'])}"
    #     f"|{V_extractor + V1 + float(values['-INNER_CYL-']):.1f}|{V_extractor + V2 + float(values['-OUTER_CYL-']):.1f}"
    #     f"|{V_extractor + float(values['-MATSUDA-']):.1f}"
    #     f"|{V_tofreference + V_reference}|{V_tofreference + V_reference + float(values['-LENS2-'])}"
    #     f"|{V_tofreference}|{V_tofextractor1}|{V_rg}")


def set_voltages_tof(values):
    """
    Set all ion_energy-independent (tof) voltages.
    """
    rv = TwTpsSetTargetValue(tps1rc['RB'], float(values['-RB-']))
    tps_error_log(rv, 'RB')
    rv = TwTpsSetTargetValue(tps1rc['TOFPULSE'], float(values['-TOFPULSE-']))
    tps_error_log(rv, 'TOFPULSE')
    rv = TwTpsSetTargetValue(tps1rc['TOFEXTR2'], float(values['-TOFEXTR2-']))
    tps_error_log(rv, 'TOFEXTR2')
    rv = TwTpsSetTargetValue(tps1rc['DRIFT'], float(values['-DRIFT-']))
    tps_error_log(rv, 'DRIFT')
    rv = TwTpsSetTargetValue(tps1rc['PA'], float(values['-PA-']))
    tps_error_log(rv, 'PA')
    rv = TwTpsSetTargetValue(tps1rc['MCP'], float(values['-MCP-']))
    tps_error_log(rv, 'MCP')
    rv = TwTpsSetTargetValue(tps1rc['HVSUPPLY'], float(values['-HVSUPPLY-']))
    tps_error_log(rv, 'HVSUPPLY')
    rv = TwTpsSetTargetValue(tps1rc['HVPOS'], float(values['-HVPOS-']))
    tps_error_log(rv, 'HVPOS')
    rv = TwTpsSetTargetValue(tps1rc['HVNEG'], float(values['-HVNEG-']))
    tps_error_log(rv, 'HVNEG')


def load_setpoints(set_file):
    """Load setpoints from file"""
    with open(set_file, 'r') as f:
        setpoints = jsonload(f)
    return setpoints


def save_setpoints(set_file, setpoints, values):
    """Save setpoints to file"""
    for key in SETPOINTS:
        setpoints[key] = values[key]
    with open(set_file, 'w') as f:
        jsondump(setpoints, f)


def read_setpoints_from_tps():
    """Read current setpoints from TPS"""
    tps2setpoint = copy.deepcopy(tps1rc)
    value = np.zeros(1, dtype=np.float64)
    for key in tps1rc:
        rv = TwTpsGetTargetValue(tps1rc[key], value)
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

    # menu_def = [['&Settings', ['&TPS IP address']], ['&Help', ['&About']]]
    menu_def = [['&Help', ['&Keyboard shortcuts...', '&Voltage mapping...', '&About...']]]
    layout = [[sg.Menu(menu_def, key='-MENU-')]]

    layout += [[sg.Frame('Energies (eV)', 
        [[sg.Text('Ion energy', size=(15,1)), sg.Input(default_text='50', size=(6,1), key='-ION_ENERGY-'),
        sg.Text('ESA energy', size=(15,1)), sg.Input(default_text='100', size=(6,1), key='-ESA_ENERGY-'), 
        sg.Text('TOF energy', size=(15,1)), sg.Input(default_text='60', size=(6,1), key='-TOF_ENERGY-'),
        # sg.Text('Polarity', size=(15,1)), sg.Combo(values=('pos', 'neg'), default_value='pos', readonly=True, key='-POLARITY-')]]
        sg.Combo(visible=False, values=('pos', 'neg'), default_value='pos', readonly=True, key='-POLARITY-')]]
        )]]

    layout += [[sg.Frame('Voltages (V)',
        [[sg.Text('Orifice', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-ORIFICE-'),
        sg.Text('Matsuda', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-MATSUDA-'),
        sg.Text('TOF Extractor 1', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-TOFEXTR1-')],
        [sg.Text('Lens 1', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-LENS1-'),
        sg.Text('Inner Cylinder', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-INNER_CYL-'),
        sg.Text('TOF Extractor 2', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-TOFEXTR2-')],
        [sg.Text('Deflector 1 up', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFL1U-'),
        sg.Text('Outer Cylinder', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-OUTER_CYL-'),
        sg.Text('TOF Pulse', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-TOFPULSE-')],
        [sg.Text('Deflector 1 down', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFL1D-'),
        sg.Text('Lens 2', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-LENS2-'),
        sg.Text('RG', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-RG-')],
        [sg.Text('Deflector 1 left', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFL1L-'),
        sg.Text('Deflector 2', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFL-'),
        sg.Text('RB', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-RB-')],
        [sg.Text('Deflector 1 right', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFL1R-'),
        sg.Text('Deflector Flange 2', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFLFL-'),
        sg.Text('Drift', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DRIFT-')],
        [sg.Text('Ion Extractor', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-IONEX-'), 
        sg.Text('Reference', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-REF-'), 
        sg.Text('PA', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-PA-')],
        [sg.Text('', size=(1,1)), 
        sg.Text('', size=(1,1)), 
        sg.Text('MCP', size=(57,1)), sg.Input(default_text='0', size=(6,1), key='-MCP-')],
        [sg.Input(visible=False, default_text='1000', key='-HVPOS-'),
        sg.Input(visible=False, default_text='-4000', key='-HVNEG-'),
        sg.Input(visible=False, default_text='1', key='-HVSUPPLY-')]]
        )]]

    layout += [[sg.Button('Send all', key='-SET_TPS-', bind_return_key=True), sg.Button('Read setpoints', key='-READ_FROM_TPS-'),  
        sg.Input(visible=False, enable_events=True, key='-LOAD-'), sg.FileBrowse('Open...', initial_folder='setpoints', target='-LOAD-', key = '-LOAD2-'), 
        sg.Input(visible=False, enable_events=True, key='-SAVE-'), sg.FileSaveAs('Save...', default_extension = 'tps', initial_folder='setpoints', target='-SAVE-', key = '-SAVE2-'),
        sg.Button('Zero all', key='-ZERO_ALL-')]]

    layout += [[sg.Frame('Scan', 
        [[sg.Text('Start ion energy (eV)', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-START_ENERGY-'),
        sg.Text('End ion energy (eV)', size=(15,1)), sg.Input(default_text='50', size=(6,1), key='-END_ENERGY-'),
        sg.Text('Step size (eV)', size=(15,1)), sg.Input(default_text='0.5', size=(6,1), key='-STEP_SIZE-')],
        [sg.Text('Time per step (s)', size=(15,1)), sg.Input(default_text='2', size=(6,1), key='-TIME_PER_STEP-')],
        [sg.Button('Start', key='-START-'), sg.Button('Cancel', key='-STOP-'),
        sg.ProgressBar(max_value=100, orientation='h', size=(20, 10), key='-PROGRESS_BAR-', expand_x=True)]]
        )]]

    layout += [[sg.Multiline(size=(50,5), autoscroll=True, 
        reroute_stdout=True, echo_stdout_stderr=True, write_only=True, key='-LOG_OUTPUT-',
        right_click_menu=['', ['&Clear']], background_color=sg.theme_background_color(), 
        text_color=sg.theme_element_text_color(), no_scrollbar=True, expand_x=True)]]

    return sg.Window('EMS scan | TOFWERK', layout, icon='tw.ico', resizable=True, finalize=True, 
        return_keyboard_events=False, enable_close_attempted_event=True)


def bind_mouse_wheel(window):
    """Bind mouse wheel to text inputs"""
    for key in SETPOINTS:
        window[key].bind('<MouseWheel>', ',+MOUSE WHEEL+')


def scanning_thread(window, values):
    """Energy scanning"""
    if not TwTofDaqRunning():
        log.error('TofDaqRec not running.')
        return

    for key, state in {'-START-': True, '-STOP-': False}.items(): window[key].update(disabled=state)

    log.info('Energy scan started.')

    progress = 0

    start_energy = float(values['-START_ENERGY-'])  # start energy, eV
    end_energy = float(values['-END_ENERGY-'])  # end energy, eV
    step_size = float(values['-STEP_SIZE-'])  # energy step stize, eV
    time_per_step = float(values['-TIME_PER_STEP-'])  # time per energy step, s

    TwTpsSaveSetFile('TwTpsTempSetFile'.encode())
    save_setpoints('./currentSetpoints.tps'.encode(), SETPOINTS, values)

    set_voltages_ea(values, start_energy)
    window['-ION_ENERGY-'].update(value=values['-START_ENERGY-'])

    # start acquisition (-> one data file per scan) 
    if TwDaqActive():
        log.warning('Stopping already running acquisition...')
        TwStopAcquisition()
        while TwDaqActive():  # wait until acquisition is stopped
            if exit_event.wait(timeout=1): break
    TwSaveIniFile(''.encode())
    TwSetDaqParameter('DataFileName'.encode(), 'EMSscan_<year>-<month>-<day>_<hour>h<minute>m<second>s.h5'.encode())
    TwStartAcquisition()
    log.info('Starting TofDaq acquisition.')
    while not TwDaqActive():  # wait until acquisition is started
        if exit_event.wait(timeout=1): break

    TwAddAttributeDouble('/EnergyData'.encode(), 'Start energy (eV)'.encode(), start_energy)
    TwAddAttributeDouble('/EnergyData'.encode(), 'End energy (eV)'.encode(), end_energy)
    TwAddAttributeDouble('/EnergyData'.encode(), 'Step size (eV)'.encode(), step_size)
    TwAddAttributeDouble('/EnergyData'.encode(), 'Time_per_step (s)'.encode(), time_per_step)
   
    # start energy scan
    log.info('Scanning...')
    for i in np.arange(start_energy, end_energy+1e-6, step_size, dtype=float):
        h5logtext = f'{i:.1f} eV'.encode()
        TwAddLogEntry(h5logtext, 0)
        set_voltages_ea(values, i)
        window['-ION_ENERGY-'].update(value=round(i, 2))
        TwUpdateUserData('/EnergyData'.encode(), 2, np.array([i, values['-ESA_ENERGY-']], dtype=np.float64))
        window['-PROGRESS_BAR-'].update_bar(progress, 100)
        progress += 100 / ((end_energy - start_energy)/step_size)
        if exit_event.wait(timeout=time_per_step): break
           
    log.info('Stopping acquisition.')
    TwStopAcquisition()
    while TwDaqActive():  # wait until acquisition is stopped
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
        log.error('TofDaqRec not running.')

    for key, state in {'-START-': False, '-STOP-': True}.items():
        window[key].update(disabled=state)

    # keyboard shortcuts
    window.bind('<Control-o>', '+OPEN+')  
    window.bind('<Control-s>', '+SAVE+')
    window.bind('<Control-z>', '+ZERO+')

    # bind mouse wheel to element keys
    bind_mouse_wheel(window)

    # Store Ion and ESA energy as a registered data source
    TwRegisterUserDataBufPy('/EnergyData', ['Ion energy (eV)', 'ESA energy (eV)'], 0)

    # Event Loop 
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSE_ATTEMPTED_EVENT:
            break
        elif event == 'About...':
            sg.popup_no_buttons('EMS scan software', 'Version ' + __version__,
                __copyright__, title = 'About', icon = 'tw.ico', image='tw.png')
        elif event == 'Keyboard shortcuts...':
            sg.popup_no_buttons(
                'Send all:       Enter', 
                'Read setpoints: Ctrl-R',
                'Open...:        Ctrl-O',
                'Save...:        Ctrl-S', 
                'Zero all:       Ctrl-Z', 
                title = 'Keyboard shortcuts', icon = 'tw.ico', font = ('Courier', 10))
        elif event == 'Voltage mapping...':
            sg.popup_no_buttons(
                'TPS_Orifice            = Orifice',
                'TPS_Lens_1             = Lens_1 + V*',
                'TPS_Deflector_1_up     = Deflector_1_up + V*',
                'TPS_Deflector_1_down   = Deflector_1_down + V*',
                'TPS_Deflector_1_left   = Deflector_1_left + V*',
                'TPS_Deflector_1_right  = Deflector_1_right + V*',
                'TPS_Ion_Extractor      = Ion_Extractor + V*',
                'TPS_Matsuda            = Matsuda + V*',
                'TPS_Inner_Cylinder     = Inner_Cylinder - 0.26706*ESA_energy*V/eV + V*',
                'TPS_Outer_Cylinder     = Outer_Cylinder + 0.23558*ESA_energy*V/eV + V*',
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
                title = 'Voltage mapping', line_width = 120, icon = 'tw.ico', font = ('Courier', 10))
        elif event == '-START-':
            threading.Thread(target=scanning_thread, args=(window,values,), daemon=True).start()
        elif event == '-STOP-':
            exit_event.set()
            log.warning('Stopping energy scan by user request.')
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
        elif event == '-SET_TPS-':
            ion_energy = float(values['-ION_ENERGY-'])
            set_voltages_ea(values, ion_energy)
            set_voltages_tof(values)
            for key in V_INPUTS:
                window[key].update(background_color='#99C794')
            TwUpdateUserData('/EnergyData'.encode(), 2, np.array([values['-ION_ENERGY-'], values['-ESA_ENERGY-']], dtype=np.float64))
            log.info('TPS voltages set.')
        elif event == '-READ_FROM_TPS-' or event == 'r:82':  # Ctrl-R
            tps2setpoint = read_setpoints_from_tps()
            rg_correction = 0.25  # ion energy correction of RG in V/eV
            tof_energy = float(values['-ION_ENERGY-']) - tps2setpoint['TOFREF']
            V_extractor = float(values['-ION_ENERGY-']) - float(values['-ESA_ENERGY-'])
            V1, V2 = calculate_EA_voltages(float(values['-ESA_ENERGY-']), polarity=1)
            window['-MCP-'].update(value=round(tps2setpoint['MCP'], 2))
            window['-PA-'].update(value=round(tps2setpoint['PA'], 2))
            window['-DRIFT-'].update(value=round(tps2setpoint['DRIFT'], 2))
            window['-TOFEXTR2-'].update(value=round(tps2setpoint['TOFEXTR2'], 2))
            window['-TOFPULSE-'].update(value=round(tps2setpoint['TOFPULSE'], 2))
            window['-RB-'].update(value=round(tps2setpoint['RB'], 2))
            window['-RG-'].update(value=round(tps2setpoint['RG'] - tps2setpoint['TOFREF']*rg_correction, 2))
            window['-ORIFICE-'].update(value=round(tps2setpoint['ORIFICE'], 2))
            window['-LENS1-'].update(value=round(tps2setpoint['L1'] - V_extractor, 2))
            window['-DEFL1U-'].update(value=round(tps2setpoint['DEFL1U'] - V_extractor, 2))
            window['-DEFL1D-'].update(value=round(tps2setpoint['DEFL1D'] - V_extractor, 2))
            window['-DEFL1R-'].update(value=round(tps2setpoint['DEFL1R'] - V_extractor, 2))
            window['-DEFL1L-'].update(value=round(tps2setpoint['DEFL1L'] - V_extractor, 2))
            window['-MATSUDA-'].update(value=round(tps2setpoint['MATSUDA'] - V_extractor, 2))
            window['-LENS2-'].update(value=round(tps2setpoint['L2'] - tps2setpoint['REFERENCE'], 2))
            window['-DEFL-'].update(value=round(tps2setpoint['DEFL'] - tps2setpoint['REFERENCE'], 2))
            window['-DEFLFL-'].update(value=round(tps2setpoint['DEFLFL'] - tps2setpoint['REFERENCE'], 2))
            window['-REF-'].update(value=round(tps2setpoint['REFERENCE'] + tof_energy - float(values['-ION_ENERGY-']), 2))
            window['-INNER_CYL-'].update(value=round(tps2setpoint['INNER_CYL'] - V1 - V_extractor, 2))
            window['-OUTER_CYL-'].update(value=round(tps2setpoint['OUTER_CYL'] - V2 - V_extractor, 2))
            window['-TOF_ENERGY-'].update(value=round(tof_energy, 2))
            window['-TOFEXTR1-'].update(value=round(tps2setpoint['TOFEXTR1'] + tof_energy - float(values['-ION_ENERGY-']), 2))
            window['-IONEX-'].update(value=round(tps2setpoint['IONEX'] - V_extractor, 2))
            log.info('Updated set values from current TPS setpoints.')
        elif event == '-ZERO_ALL-' or event == '+ZERO+':  # Ctrl-z
            zero_all()
            for key in V_INPUTS:
                window[key].update(background_color='#6699CC')
            log.info('All voltages set to zero.')
        # elif re.search('\+MOUSE WHEEL\+$', event) is not None:
        elif event.endswith('+MOUSE WHEEL+'):
            key = re.split(',', event)[0]
            window[key].update(value=round(float(values[key]) - float(window[key].user_bind_event.delta/120), 2))

    save_setpoints('./currentSetpoints.tps'.encode(), SETPOINTS, values)
    TwUnregisterUserData('/EnergyData'.encode())
    TwTpsDisconnect()
    TwCleanupDll()

    window.close()
    sys.exit(0)

if __name__ == '__main__':
    main()
