#! /usr/bin/python3-64

"""Basic EMS voltage control and energy scanning"""

__version__ = '0.1.6'
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
'EP': 19,
'REFERENCE': 117,
'ORIFICE': 2500,
'INNER_CYL': 2501,
'OUTER_CYL': 2502,
'MATSUDA': 2503,
'DEFL2U': 2504,
'DEFL2D': 2505,
'DEFL2L': 2506,
'DEFL2R': 2507,
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
V_INPUTS = {'-ORIFICE-':0, '-LENS1-':0, '-DEFL2U-':0, '-DEFL2D-':0, '-DEFL2L-':0, '-DEFL2R-':0, 
    '-INNER_CYL-':0, '-OUTER_CYL-':0, '-MATSUDA-':0, '-LENS2-':0, '-DEFL-':0, '-DEFLFL-':0, '-REF-':0,
    '-TOFEXTR1-':0, '-RG-':0, '-RB-':0, '-TOFEXTR2-':0, '-TOFPULSE-':0, '-DRIFT-':0,
    '-PA-':0, '-MCP-':0}

# Window element keys that will be saved to a file
SETPOINTS = {'-ESA_ENERGY-':0, '-TOF_ENERGY-':0, '-ION_ENERGY-':0, '-POLARITY-':0, 
    '-ORIFICE-':0, '-LENS1-':0, '-DEFL2U-':0, '-DEFL2D-':0, '-DEFL2L-':0, '-DEFL2R-':0, 
    '-INNER_CYL-':0, '-OUTER_CYL-':0, '-MATSUDA-':0, '-LENS2-':0, '-DEFL-':0, '-DEFLFL-':0, '-REF-':0,
    '-START_ENERGY-':0, '-END_ENERGY-':0, '-STEP_SIZE-':0,'-TIME_PER_STEP-':0,
    '-TOFEXTR1-':0, '-RG-':0, '-RB-':0, '-TOFEXTR2-':0, '-TOFPULSE-':0, '-DRIFT-':0,
    '-PA-':0, '-MCP-':0, '-HVSUPPLY-':0, '-HVPOS-':0, '-HVNEG-':0}


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


def set_voltages_ea(values, ion_energy):
    """
    Set all ion_energy-dependent voltages.
    """ 
    # polarity = 1 if (values['-POLARITY-']=='pos') else -1  # Note: does not yet work for neg
    V1, V2 = calculate_EA_voltages(float(values['-ESA_ENERGY-']), polarity=1)
    V_extractor = ion_energy - float(values['-ESA_ENERGY-'])
    V_reference = float(values['-REF-'])
    V_tofreference = ion_energy - float(values['-TOF_ENERGY-'])  # from LV channel -> with sign
    V_tofextractor1 = ion_energy - float(values['-TOF_ENERGY-']) + float(values['-TOFEXTR1-'])  # from LV channel -> with sign, relative to TOF reference
    rg_correction = 0.25  # ion energy correction of RG in V/eV
    V_rg = float(values['-RG-']) + ion_energy*rg_correction  # -RG- is set value at 0 eV ion energy

    rv = TwTpsSetTargetValue(tps1rc['ORIFICE'], float(values['-ORIFICE-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['ORIFICE']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['IONEX'], V_extractor)
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['IONEX']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['L1'], V_extractor + float(values['-LENS1-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['L1']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['DEFL2U'], V_extractor + float(values['-DEFL2U-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['DEFL2U']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['DEFL2D'], V_extractor + float(values['-DEFL2D-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['DEFL2D']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['DEFL2R'], V_extractor + float(values['-DEFL2R-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['DEFL2R']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['DEFL2L'], V_extractor + float(values['-DEFL2L-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['DEFL2L']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['INNER_CYL'], V_extractor + V1 + float(values['-INNER_CYL-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['INNER_CYL']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['OUTER_CYL'], V_extractor + V2 + float(values['-OUTER_CYL-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['OUTER_CYL']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['MATSUDA'], V_extractor + float(values['-MATSUDA-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['MATSUDA']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['REFERENCE'], V_tofreference + V_reference)
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['REFERENCE']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['L2'], V_tofreference + V_reference + float(values['-LENS2-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['L2']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['DEFL'], V_tofreference + V_reference + float(values['-DEFL-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['DEFL']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['DEFLFL'], V_tofreference + V_reference + float(values['-DEFLFL-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['DEFLFL']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['TOFREF'], V_tofreference)
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['TOFREF']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['TOFEXTR1'], V_tofextractor1)
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['TOFEXTR1']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['RG'], V_rg)
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['RG']}: {TwTranslateReturnValue(rv).decode()}.")

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
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['RB']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['TOFPULSE'], float(values['-TOFPULSE-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['TOFPULSE']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['TOFEXTR2'], float(values['-TOFEXTR2-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['TOFEXTR2']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['DRIFT'], float(values['-DRIFT-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['DRIFT']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['PA'], float(values['-PA-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['PA']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['MCP'], float(values['-MCP-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['MCP']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['HVSUPPLY'], float(values['-HVSUPPLY-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['HVSUPPLY']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['HVPOS'], float(values['-HVPOS-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['HVPOS']}: {TwTranslateReturnValue(rv).decode()}.")
    rv = TwTpsSetTargetValue(tps1rc['HVNEG'], float(values['-HVNEG-']))
    if (rv != TwSuccess): log.error(f"Failed to set value for RC code {tps1rc['HVNEG']}: {TwTranslateReturnValue(rv).decode()}.")


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


def make_window():
    """Make GUI window"""
    sg.SetOptions(text_justification='right')

    menu_def = [['&Settings', ['&TPS IP address']], ['&Help', ['&About']]]
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
        [sg.Text('Defl 2 up', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFL2U-'),
        sg.Text('Outer Cylinder', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-OUTER_CYL-'),
        sg.Text('TOF Pulse', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-TOFPULSE-')],
        [sg.Text('Defl 2 down', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFL2D-'),
        sg.Text('Lens 2', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-LENS2-'),
        sg.Text('RG', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-RG-')],
        [sg.Text('Defl 2 left', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFL2L-'),
        sg.Text('Defl 2', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFL-'),
        sg.Text('RB', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-RB-')],
        [sg.Text('Defl 2 right', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFL2R-'),
        sg.Text('Defl Fl 2', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFLFL-'),
        sg.Text('Drift', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DRIFT-')],
        [sg.Text('', size=(1,1)), 
        sg.Text('Reference', size=(36,1)), sg.Input(default_text='0', size=(6,1), key='-REF-'), 
        sg.Text('PA', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-PA-')],
        [sg.Text('', size=(1,1)), 
        sg.Text('', size=(1,1)), 
        sg.Text('MCP', size=(57,1)), sg.Input(default_text='0', size=(6,1), key='-MCP-')],
        [sg.Input(visible=False, default_text='1000', key='-HVPOS-'),
        sg.Input(visible=False, default_text='-4000', key='-HVNEG-'),
        sg.Input(visible=False, default_text='1', key='-HVSUPPLY-')]]
        )]]

    layout += [[sg.Button('Send all', key='-SET_TPS-'), sg.Button('Read setpoints', key='-READ_FROM_TPS-'), 
        sg.Input(visible=False, enable_events=True, do_not_clear=False, key='-LOAD-'), sg.FilesBrowse('Open...', initial_folder='setpoints', target='-LOAD-'), 
        sg.Input(visible=False, enable_events=True, do_not_clear=False, key='-SAVE-'), sg.FileSaveAs('Save...', default_extension = 'tps', initial_folder='setpoints', target='-SAVE-'),
        sg.Button('Zero all', key='-ZERO_ALL-')]]

    layout += [[sg.Frame('Scan', 
        [[sg.Text('Start ion energy (eV)', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-START_ENERGY-'),
        sg.Text('End ion energy (eV)', size=(15,1)), sg.Input(default_text='100', size=(6,1), key='-END_ENERGY-'),
        sg.Text('Step size (eV)', size=(15,1)), sg.Input(default_text='10', size=(6,1), key='-STEP_SIZE-')],
        [sg.Text('Time per step (s)', size=(15,1)), sg.Input(default_text='1', size=(6,1), key='-TIME_PER_STEP-')],
        [sg.Button('Start', key='-START-'), sg.Button('Cancel', key='-STOP-'),
        sg.ProgressBar(max_value=100, orientation='h', size=(41, 10), key='-PROGRESS BAR-')]]
        )]]

    layout += [sg.Text('Log')], [sg.Multiline(size=(80,10), autoscroll=True, 
        reroute_stdout=True, echo_stdout_stderr=True, write_only=True, key='-LOG_OUTPUT-',
        right_click_menu=['', ['&Clear']])]
       
    return sg.Window('EMS scan | TOFWERK', layout, icon='tw.ico', resizable=True, finalize=True)


def scanning_thread(window, values):
    """Energy scanning"""
    progress = 0

    start_energy = float(values['-START_ENERGY-'])  # start energy, eV
    end_energy = float(values['-END_ENERGY-'])  # end energy, eV
    step_size = float(values['-STEP_SIZE-'])  # energy step stize, eV
    time_per_step = float(values['-TIME_PER_STEP-'])  # time per energy step, s

    # TwTpsSaveSetFile('TwTpsTempSetFile'.encode())
    set_voltages_ea(values, start_energy)
    window['-ION_ENERGY-'].update(value=values['-START_ENERGY-'])

    # start acquisition (-> one data file per scan) 
    if TwDaqActive():
        log.warning('Stopping already running acquisition...')
        TwStopAcquisition()
        if exit_event.wait(timeout=5): exit_event.set()
    TwSaveIniFile(''.encode())
    TwSetDaqParameter('DataFileName'.encode(), 'EMSscan_<year>-<month>-<day>_<hour>h<minute>m<second>s.h5'.encode())
    TwStartAcquisition()
    log.info('Starting TofDaq acquisition.')
    if exit_event.wait(timeout=1): exit_event.set()

    TwAddAttributeDouble('/'.encode(), 'start energy (eV)'.encode(), start_energy)
    TwAddAttributeDouble('/'.encode(), 'end energy (eV)'.encode(), end_energy)
    TwAddAttributeDouble('/'.encode(), 'step size (eV)'.encode(), step_size)
    TwAddAttributeDouble('/'.encode(), 'time_per_step (s)'.encode(), time_per_step)

    # start energy scan
    log.info('Scanning...')
    for i in np.arange(start_energy, end_energy+1e-6, step_size, dtype=float):
        h5logtext = f'{i:.1f} eV'.encode()
        TwAddLogEntry(h5logtext, 0)
        set_voltages_ea(values, i)
        window['-ION_ENERGY-'].update(value=round(i, 2))
        if exit_event.wait(timeout=time_per_step): break
        progress += 100 / ((end_energy - start_energy)/step_size)
        window.write_event_value('-PROGRESS-', progress)

    log.info('Stopping acquisition.')
    TwStopAcquisition()
    time.sleep(1)
    TwLoadIniFile(''.encode())
    # TwTpsLoadSetFile('TwTpsTempSetFile'.encode())
    log.info('Fertig.')
    [window[key].update(disabled=value) for key, value in {'-START-': False, '-STOP-': True}.items()]


def main():
    window = make_window()
    
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

    # Event Loop 
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == 'About':
            sg.popup('EMS scan software', 'Version ' + __version__,
                __copyright__, title = 'About', icon = 'tw.ico', image='tw.png')
        elif event == '-PROGRESS-':
            window['-PROGRESS BAR-'].update_bar(values[event], 100)
        elif event == 'TPS IP address':
            new_tps_ip=sg.popup_get_text('TPS IP address', default_text=tps_ip, size=(15,1), icon='tw.ico')
            if new_tps_ip!=None:
                tps_ip = new_tps_ip
                rv = TwTpsConnect2(new_tps_ip.encode(), 1)
                if rv != TwSuccess:
                    log.error('Failed to connect to TPS2.')
                else:
                    log.info(f'TPS2 connected via {tps_ip}.')
                    # TwTpsSaveSetFile('TwTpsTempSetFile'.encode())
        elif event == '-START-':
            for key, state in {'-START-': True, '-STOP-': False}.items():
                window[key].update(disabled=state)
            log.info('Energy scan started.')
            threading.Thread(target=scanning_thread, args=(window,values,), daemon=True).start()
        elif event == '-STOP-':
            # [window[key].update(disabled=value) for key, value in {'-START-': False, '-STOP-': True}.items()]
            exit_event.set()
            exit_event.clear()
            log.warning('Stopped energy scan by user request.')
        elif event == 'Clear':
            window['-LOG_OUTPUT-'].update('')
        elif event == '-SAVE-':
            if values[event]!='':
                save_setpoints(values[event], setpoints, values)
                log.info(f'Set values saved to {os.path.basename(values[event])}')
        elif event == '-LOAD-':
            if values[event]!='':
                setpoints=load_setpoints(values[event])
                for key in SETPOINTS:
                    window[key].update(value=setpoints[key])
                log.info(f'Set values loaded from {os.path.basename(values[event])}')
        elif event == '-SET_TPS-':
            ion_energy = float(values['-ION_ENERGY-'])
            set_voltages_ea(values, ion_energy)
            set_voltages_tof(values)
            for key in V_INPUTS:
                window[key].update(background_color='#99C794')
            log.info('TPS voltages set.')
        elif event == '-READ_FROM_TPS-':
            tps2setpoint = read_setpoints_from_tps()
            rg_correction = 0.25  # ion energy correction of RG in V/eV
            tof_energy = float(values['-ION_ENERGY-']) - tps2setpoint['TOFREF']
            V_extractor = float(values['-ION_ENERGY-'])-float(values['-ESA_ENERGY-'])
            V1, V2 = calculate_EA_voltages(float(values['-ESA_ENERGY-']), polarity=1)
            window['-MCP-'].update(value=tps2setpoint['MCP'])
            window['-PA-'].update(value=tps2setpoint['PA'])
            window['-DRIFT-'].update(value=tps2setpoint['DRIFT'])
            window['-TOFEXTR2-'].update(value=tps2setpoint['TOFEXTR2'])
            window['-TOFPULSE-'].update(value=tps2setpoint['TOFPULSE'])
            window['-RB-'].update(value=tps2setpoint['RB'])
            window['-RG-'].update(value=tps2setpoint['RG'] - float(values['-ION_ENERGY-'])*rg_correction)
            window['-ORIFICE-'].update(value=tps2setpoint['ORIFICE'])
            window['-LENS1-'].update(value=tps2setpoint['L1'] - V_extractor)
            window['-DEFL2U-'].update(value=tps2setpoint['DEFL2U'] - V_extractor)
            window['-DEFL2D-'].update(value=tps2setpoint['DEFL2D'] - V_extractor)
            window['-DEFL2R-'].update(value=tps2setpoint['DEFL2R'] - V_extractor)
            window['-DEFL2L-'].update(value=tps2setpoint['DEFL2L'] - V_extractor)
            window['-MATSUDA-'].update(value=round(tps2setpoint['MATSUDA'] - V_extractor, 2))
            window['-LENS2-'].update(value=tps2setpoint['L2'] - tps2setpoint['REFERENCE'])
            window['-DEFL-'].update(value=tps2setpoint['DEFL'] - tps2setpoint['REFERENCE'])
            window['-DEFLFL-'].update(value=tps2setpoint['DEFLFL'] - tps2setpoint['REFERENCE'])
            window['-REF-'].update(value=tps2setpoint['REFERENCE'] + tof_energy - float(values['-ION_ENERGY-']))
            window['-INNER_CYL-'].update(value=round(tps2setpoint['INNER_CYL'] - V1 - V_extractor, 2))
            window['-OUTER_CYL-'].update(value=round(tps2setpoint['OUTER_CYL'] - V2 - V_extractor, 2))
            window['-TOF_ENERGY-'].update(value=tof_energy)
            window['-TOFEXTR1-'].update(value=tps2setpoint['TOFEXTR1'] + tof_energy - float(values['-ION_ENERGY-']))
            log.info('Updated set values from current TPS setpoints.')
        elif event == '-ZERO_ALL-':
            zero_all()
            for key in V_INPUTS:
                window[key].update(background_color='#6699CC')
            log.info('All voltages set to zero.')

    TwTpsDisconnect()
    TwCleanupDll()

    window.close()
    sys.exit(0)

if __name__ == '__main__':
    main()
