#! /usr/bin/python3-64

"""Basic EMS voltage control and energy scanning"""

__version__ = '0.1.1'
__author__ = 'Patrick Sturm'
__copyright__ = 'Copyright 2021, TOFWERK'

import numpy as np
import time
import sys
import math
import logging
import threading
import os
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
'ORIFICE': -1,
'INNER_CYL': -1,
'OUTER_CYL': -1,
'MATSUDA': -1,
'DEFL1U': -1,
'DEFL1D': -1,
'DEFL1R': -1,
'DEFL1L': -1,
'TOFREF': 202,
'TOFEXTR1': 201,
'RG': 2
}


# Window element keys that will be saved to a file
SETPOINTS = {'-ESA_ENERGY-':0, '-TOF_ENERGY-':0, '-ION_ENERGY-':0, '-POLARITY-':0, 
    '-ORIFICE-':0, '-LENS1-':0, '-DEFL1U-':0, '-DEFL1D-':0, '-DEFL1L-':0, '-DEFL1R-':0, 
    '-ESA_OFFSET-':0, '-MATSUDA-':0, '-LENS2-':0, '-DEFL2-':0, '-DEFLFL2-':0, '-REF-':0,
    '-START_ENERGY-':0, '-END_ENERGY-':0, '-STEP_SIZE-':0,'-TIME_PER_STEP-':0,
    '-TOFEXTR1-':0, '-RG-':0}


# exit event to abort energy scanning
exit_event = threading.Event()


# Function definitions --------------------------------------------------------
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


def set_voltages(values, ion_energy):
    """
    Set all voltages.
    """
    polarity = 1 if (values['-POLARITY-']=='pos') else -1
    V1, V2 = calculate_EA_voltages(float(values['-ESA_ENERGY-']), polarity=polarity)
    V_extractor = (float(values['-ESA_ENERGY-'])-ion_energy)*(-polarity)
    V_reference = float(values['-REF-'])
    V_tofreference = (float(values['-TOF_ENERGY-'])-ion_energy)*(-polarity)  # from LV channel -> with sign
    V_tofextractor1 = (float(values['-TOF_ENERGY-'])-ion_energy + float(values['-TOFEXTR1-']))*(-polarity)  # from LV channel -> with sign, relative to TOF reference
    rg_correction = 0.25  # ion energy correction of RG in V/eV
    V_rg = float(values['-RG-']) + ion_energy*rg_correction  # -RG- is set value at 0 eV ion energy
    

    # rv = TwTpsSetTargetValue(tps1rc['ORIFICE'], float(values['-ORIFICE-']))
    # if (rv != TwSuccess): log.error(f'Failed to set value for RC code {tps1rc["ORIFICE"]}: {TwTranslateReturnValue(rv).decode("utf-8")}.')
    # rv = TwTpsSetTargetValue(tps1rc['IONEX'], V_extractor)
    # if (rv != TwSuccess): log.error(f'Failed to set value for RC code {tps1rc["IONEX"]}: {TwTranslateReturnValue(rv).decode("utf-8")}.')
    # rv = TwTpsSetTargetValue(tps1rc['L1'], V_extractor + float(values['-LENS1-']))
    # if (rv != TwSuccess): log.error(f'Failed to set value for RC code {tps1rc["L1"]}: {TwTranslateReturnValue(rv).decode("utf-8")}.')
    # rv = TwTpsSetTargetValue(tps1rc['DEFL1U'], V_extractor + float(values['-DEFL1U-']))
    # if (rv != TwSuccess): log.error(f'Failed to set value for RC code {tps1rc["DEFL1U"]}: {TwTranslateReturnValue(rv).decode("utf-8")}.')
    # rv = TwTpsSetTargetValue(tps1rc['DEFL1D'], V_extractor + float(values['-DEFL1D-']))
    # if (rv != TwSuccess): log.error(f'Failed to set value for RC code {tps1rc["DEFL1D"]}: {TwTranslateReturnValue(rv).decode("utf-8")}.')
    # rv = TwTpsSetTargetValue(tps1rc['DEFL1R'], V_extractor + float(values['-DEFL1R-']))
    # if (rv != TwSuccess): log.error(f'Failed to set value for RC code {tps1rc["DEFL1R"]}: {TwTranslateReturnValue(rv).decode("utf-8")}.')
    # rv = TwTpsSetTargetValue(tps1rc['DEFL1L'], V_extractor + float(values['-DEFL1L-']))
    # if (rv != TwSuccess): log.error(f'Failed to set value for RC code {tps1rc["DEFL1L"]}: {TwTranslateReturnValue(rv).decode("utf-8")}.')
    # rv = TwTpsSetTargetValue(tps1rc['INNER_CYL'], V_extractor + V1 + float(values['-ESA_OFFSET-']))
    # if (rv != TwSuccess): log.error(f'Failed to set value for RC code {tps1rc["INNER_CYL"]}: {TwTranslateReturnValue(rv).decode("utf-8")}.')
    # rv = TwTpsSetTargetValue(tps1rc['OUTER_CYL'], V_extractor + V2 + float(values['-ESA_OFFSET-']))
    # if (rv != TwSuccess): log.error(f'Failed to set value for RC code {tps1rc["OUTER_CYL"]}: {TwTranslateReturnValue(rv).decode("utf-8")}.')
    # rv = TwTpsSetTargetValue(tps1rc['MATSUDA'], V_extractor + float(values['-MATSUDA-']))
    # if (rv != TwSuccess): log.error(f'Failed to set value for RC code {tps1rc["MATSUDA"]}: {TwTranslateReturnValue(rv).decode("utf-8")}.')
    # rv = TwTpsSetTargetValue(tps1rc['REFERENCE'], V_tofreference + V_reference)
    # if (rv != TwSuccess): log.error(f'Failed to set value for RC code {tps1rc["REFERENCE"]}: {TwTranslateReturnValue(rv).decode("utf-8")}.')
    # rv = TwTpsSetTargetValue(tps1rc['L2'], V_tofreference + V_reference + float(values['-LENS2-']))
    # if (rv != TwSuccess): log.error(f'Failed to set value for RC code {tps1rc["L2"]}: {TwTranslateReturnValue(rv).decode("utf-8")}.')
    # rv = TwTpsSetTargetValue(tps1rc['DEFL'], V_tofreference + V_reference + float(values['-DEFL2-']))
    # if (rv != TwSuccess): log.error(f'Failed to set value for RC code {tps1rc["DEFL"]}: {TwTranslateReturnValue(rv).decode("utf-8")}.')
    # rv = TwTpsSetTargetValue(tps1rc['DEFLFL'], V_tofreference + V_reference + float(values['-DEFLFL2-']))
    # if (rv != TwSuccess): log.error(f'Failed to set value for RC code {tps1rc["DEFLFL"]}: {TwTranslateReturnValue(rv).decode("utf-8")}.')
    # rv = TwTpsSetTargetValue(tps1rc['TOFREF'], V_tofreference)
    # if (rv != TwSuccess): log.error(f'Failed to set value for RC code {tps1rc["TOFREF"]}: {TwTranslateReturnValue(rv).decode("utf-8")}.')
    # rv = TwTpsSetTargetValue(tps1rc['TOFEXTR1'], V_tofextractor1)
    # if (rv != TwSuccess): log.error(f'Failed to set value for RC code {tps1rc["TOFEXTR1"]}: {TwTranslateReturnValue(rv).decode("utf-8")}.')
    # rv = TwTpsSetTargetValue(tps1rc['RG'], V_rg)
    # if (rv != TwSuccess): log.error(f'Failed to set value for RC code {tps1rc["RG"]}: {TwTranslateReturnValue(rv).decode("utf-8")}.')

    # log.debug(f"Orifice {values['-ORIFICE-']}|Extractor {V_extractor}|L1 {V_extractor + float(values['-LENS1-'])}"
    #     f"|InnerCyl {V_extractor + V1:.1f}|OuterCyl {V_extractor + V2:.1f}|Matsuda {V_extractor + float(values['-MATSUDA-']):.1f}"
    #     f"|Reference {V_reference}|L2 {V_reference + float(values['-LENS2-'])}")
    # Show actual TPS voltages as debug message: Orifice|Extractor|Lens1|Inner|Outer|MaV_tofextractor1tsuda|Reference|Lens2|TOFreference|TOFExtr1|RG
    log.debug(f"{values['-ORIFICE-']}|{V_extractor}|{V_extractor + float(values['-LENS1-'])}"
        f"|{V_extractor + V1 + float(values['-ESA_OFFSET-']):.1f}|{V_extractor + V2 + float(values['-ESA_OFFSET-']):.1f}"
        f"|{V_extractor + float(values['-MATSUDA-']):.1f}"
        f"|{V_tofreference + V_reference}|{V_tofreference + V_reference + float(values['-LENS2-'])}"
        f"|{V_tofreference}|{V_tofextractor1}|{V_rg}")


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


def make_window():
    """Make GUI window"""
    sg.SetOptions(text_justification='right')

    menu_def = [['&Settings', ['&TPS IP']], ['&Help', ['&About']]]
    layout = [[sg.Menu(menu_def, key='-MENU-')]]

    layout += [[sg.Frame('Energies (eV)', 
        [[sg.Text('ESA energy', size=(15,1)), sg.Input(default_text='100', size=(6,1), key='-ESA_ENERGY-'), 
        sg.Text('TOF energy', size=(15,1)), sg.Input(default_text='60', size=(6,1), key='-TOF_ENERGY-')],
        [sg.Text('Ion energy', size=(15,1)), sg.Input(default_text='50', size=(6,1), key='-ION_ENERGY-'),
        sg.Text('Polarity', size=(15,1)), sg.Combo(values=('pos', 'neg'), default_value='pos', readonly=True, key='-POLARITY-')]]
        )]]

    layout += [[sg.Frame('Voltages (V)', 
        [[sg.Text('Orifice', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-ORIFICE-'),
        sg.Text('Matsuda', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-MATSUDA-')],
        [sg.Text('Lens 1', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-LENS1-'),
        sg.Text('ESA offset', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-ESA_OFFSET-')],
        [sg.Text('Defl 1 up', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFL1U-'),
        sg.Text('Lens 2', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-LENS2-')],
        [sg.Text('Defl 1 down', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFL1D-'),
        sg.Text('Defl 2', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFL2-')],
        [sg.Text('Defl 1 left', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFL1L-'),
        sg.Text('Defl Fl 2', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFLFL2-')],
        [sg.Text('Defl 1 right', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-DEFL1R-'),
        sg.Text('Reference', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-REF-')],
        [sg.Text('TOF Extractor 1', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-TOFEXTR1-'),
        sg.Text('RG', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-RG-')]]
        )]]

    layout += [[sg.Button('Set values', key='-SET_TPS-'), 
        sg.Input(visible=False, enable_events=True, do_not_clear=False, key='-LOAD-'), sg.FilesBrowse('Open...', initial_folder='setpoints', target='-LOAD-'), 
        sg.Input(visible=False, enable_events=True, do_not_clear=False, key='-SAVE-'), sg.FileSaveAs('Save...', default_extension = 'set', initial_folder='setpoints', target='-SAVE-')]]

    layout += [[sg.Frame('Scan', 
        [[sg.Text('Start ion energy (eV)', size=(15,1)), sg.Input(default_text='0', size=(6,1), key='-START_ENERGY-'),
        sg.Text('End ion energy (eV)', size=(15,1)), sg.Input(default_text='100', size=(6,1), key='-END_ENERGY-')],
        [sg.Text('Step size (eV)', size=(15,1)), sg.Input(default_text='10', size=(6,1), key='-STEP_SIZE-'),
        sg.Text('Time per step (s)', size=(15,1)), sg.Input(default_text='1', size=(6,1), key='-TIME_PER_STEP-')],
        [sg.Button('Start', key='-START-'), sg.Button('Cancel', key='-STOP-'),
        sg.ProgressBar(max_value=100, orientation='h', size=(24, 10), key='-PROGRESS BAR-')]]
        )]]

    layout += [[sg.Text('Log')], [sg.Multiline(size=(65,10), autoscroll=True, 
        reroute_stdout=True, echo_stdout_stderr=True, write_only=True, key='-LOG_OUTPUT-',
        right_click_menu=['', ['&Clear']])]]
       
    return sg.Window('EMS control | TOFWERK', layout, icon='tw.ico', resizable=True, finalize=True)


def scanning_thread(window, values):
    """Energy scanning"""
    progress = 0

    start_energy = float(values['-START_ENERGY-'])  # start energy, eV
    end_energy = float(values['-END_ENERGY-'])  # end energy, eV
    step_size = float(values['-STEP_SIZE-'])  # energy step stize, eV
    time_per_step = float(values['-TIME_PER_STEP-'])  # time per energy step, s

    # TwTpsSaveSetFile('TwTpsTempSetFile'.encode('utf-8'))
    set_voltages(values, start_energy)
    window['-ION_ENERGY-'].update(value=values['-START_ENERGY-'])

    # start acquisition (-> one data file per scan) 
    if TwDaqActive():
        log.warning('Stopping already running acquisition...')
        TwStopAcquisition()
        if exit_event.wait(timeout=5): exit_event.set()
    TwSaveIniFile(''.encode('utf-8'))
    TwSetDaqParameter('DataFileName'.encode('utf-8'), 'EMSscan_<year>-<month>-<day>_<hour>h<minute>m<second>s.h5'.encode('utf-8'))
    TwStartAcquisition()
    log.info('Starting TofDaq acquisition.')
    if exit_event.wait(timeout=1): exit_event.set()

    TwAddAttributeDouble('/'.encode('utf-8'), 'start energy (eV)'.encode('utf-8'), start_energy)
    TwAddAttributeDouble('/'.encode('utf-8'), 'end energy (eV)'.encode('utf-8'), end_energy)
    TwAddAttributeDouble('/'.encode('utf-8'), 'step size (eV)'.encode('utf-8'), step_size)
    TwAddAttributeDouble('/'.encode('utf-8'), 'time_per_step (s)'.encode('utf-8'), time_per_step)

    # start energy scan
    log.info('Scanning...')
    for i in np.arange(start_energy, end_energy+1e-6, step_size, dtype=float):
        h5logtext = f'{i:.1f} eV'.encode('utf-8')
        TwAddLogEntry(h5logtext, 0)
        set_voltages(values, i)
        window['-ION_ENERGY-'].update(value=i)
        if exit_event.wait(timeout=time_per_step): break
        progress += 100 / ((end_energy - start_energy)/step_size)
        window.write_event_value('-PROGRESS-', progress)

    log.info('Stopping acquisition.')
    TwStopAcquisition()
    time.sleep(1)
    TwLoadIniFile(''.encode('utf-8'))
    # TwTpsLoadSetFile('TwTpsTempSetFile'.encode('utf-8'))
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
        rv = TwTpsConnect2(tps_ip.encode('utf-8'), 1)
        if rv != TwSuccess:
            log.error('Failed to connect to TPS2.')
        else:
            log.info(f'TPS2 connected via {tps_ip}.')
            # TwTpsSaveSetFile('TwTpsTempSetFile'.encode('utf-8'))
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
            sg.popup('EMS control software', 'Version ' + __version__,
                __copyright__, title = 'About', icon = 'tw.ico', image='tw.png')
        elif event == '-PROGRESS-':
            window['-PROGRESS BAR-'].update_bar(values[event], 100)
        elif event == 'TPS IP':
            new_tps_ip=sg.popup_get_text('TPS IP', default_text=tps_ip, size=(15,1), icon='tw.ico')
            if new_tps_ip!=None:
                tps_ip = new_tps_ip
                rv = TwTpsConnect2(new_tps_ip.encode('utf-8'), 1)
                if rv != TwSuccess:
                    log.error('Failed to connect to TPS2.')
                else:
                    log.info(f'TPS2 connected via {tps_ip}.')
                    # TwTpsSaveSetFile('TwTpsTempSetFile'.encode('utf-8'))
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
                log.info(f'Set values saved to {os.path.basename(values[event])}.')
        elif event == '-LOAD-':
            if values[event]!='':
                setpoints=load_setpoints(values[event])
                for key in SETPOINTS:
                    window[key].update(value=setpoints[key])
                log.info(f'Set values loaded from {os.path.basename(values[event])}.')
        elif event == '-SET_TPS-':
            ion_energy = float(values['-ION_ENERGY-'])
            set_voltages(values, ion_energy)
            log.info('TPS voltages set.')

    TwTpsDisconnect()
    TwCleanupDll()

    window.close()
    sys.exit(0)

if __name__ == '__main__':
    main()
