import os
import glob
import sys
import matplotlib.pyplot as plt
import numpy as np


mappings = {'Duplex-Receptacle-15A': 1, 'Hard-Wired-Connection': 2, 'Data-Outlet-Wall-Mount': 3,
            'Combination-1V-1D-Outlet-Wall-Mount': 4, 'Wall-Mount-Luminaire': 5, 'Surface-Ceiling-Mount-Luminaire': 6,
            'Voice-Outlet-Wall-Mount': 7, 'Panelboard-Surface-Mount': 8, 'Switch-Single-Pole': 9,
            'Motor-Integral-Disconnect': 10, 'Motor': 11, 'GFCI-Receptacle-15A': 12, 'Quad-Receptacle-15A': 13,
            'Disconnect-Switch-WP': 14, 'Remote-DC-1-Head': 15, '2x2-Drop-In-Fixture': 16, '2x2-Recessed-Fixture': 17,
            '2x4-Drop-In-Fixture': 18, "4'-Strip-Light-Emergency-Circuit": 19, 'Beacon-Strobe-Public-Address': 20,
            'Bollard-Light': 21, 'Buzzer-Autotransformer': 22, 'CCTV-Exterior-WP-Camera': 23,
            'CCTV-Interior-Camera': 24, 'Car-Pedestal': 25, 'Car-Pedestal-Split-Quad-Receptacle-15A': 26,
            'Car-Pedestal-Split-Receptacle-15A': 27, 'Card-Reader': 28, 'Card-Reader-Keypad': 29,
            'Ceiling-Junction-Box': 30, 'Ceiling-Junction-Slab-Box': 31, 'Clock-Wall-Mount': 32,
            'Combination-1V-1D-Outlet': 33, 'Combination-1V-1D-Outlet-Counter-Height': 34,
            'Combination-1V-1D-Outlet-Floor-Mount': 35, 'Combination-Exit-DC-Heads': 36, 'Contactor-Power-Lighting': 37,
            'Data-Outlet-2-Port': 38, 'Data-Outlet-Ceiling-Mount': 39, 'Data-Outlet-Counter-Height': 40,
            'Data-Outlet-Floor-Mount': 41, 'Daylight-Sensor': 42, 'Disconnect-Switch-Fused': 43,
            'Disconnect-Switch-Unfused': 44, 'Door-Contact': 45, 'Door-Electric-Strike': 46, 'Door-Magnetic-Lock': 47,
            'Door-Operator-Direct-Connection': 48, 'Duct-Smoke-Detector-Addressable': 49,
            'Duct-Smoke-Shut-Down-Relay': 50, 'Duplex-Receptacle-15A-WP': 51, 'Duplex-Receptacle-20A-Tslot': 52,
            'Duplex-Receptacle-Ceiling-Mount-15A': 53, 'Duplex-Receptacle-Floor-Mount-15A': 54,
            'Duplex-Receptacle-Isolated-Ground': 55, 'Duplex-Receptacle-Split-15A': 56,
            'Duplex-Receptacle-Switched-15A': 57, 'Duplex-Receptacle-TR': 58, 'Duplex-Receptacle-USB': 59,
            'Emergency-Battery-Unit': 60, 'Emergency-Battery-Unit-CW-DC-Heads': 61, 'Emergency-Call-Button': 62,
            'Emergency-Stop-Button': 63, 'End-Of-Line-Resistor': 64, 'Exit-Light': 65, 'Exit-Light-Ceiling-Mount': 66,
            'Exit-Light-Wall-Mount': 67, 'Exterior-Strobe-Public-Address': 68, 'Exterior-Wall-Pack': 69,
            'Fire-Alarm-Active-Graphic': 70, 'Fire-Alarm-Control-Panel': 71, 'Fire-Alarm-Passive-Graphic': 72,
            'Floor-Junction-Slab-Box': 73, 'GFCI-Receptacle-20A-Tslot': 74, 'GFCI-Receptacle-TR-15A': 75,
            'Glass-Break-Sensor': 76, 'Heat-Detector-Addressable': 77, 'Heat-Detector-Fixed-58C-Addressable': 78,
            'Heat-Detector-Fixed-88C-Addressable': 79, 'Heat-Detector-RateofRise-Addressable': 80,
            'Heat-Smoke-Combination-Detector': 81, 'Horn-FA-Ceiling-Mount': 82, 'Horn-FA-Wall-Mount': 83,
            'Horn-Strobe-FA': 84, 'Horn-Strobe-FA-Ceiling-Mount': 85, 'Horn-Strobe-FA-Wall-Mount': 86,
            'Humidistat-Mech-Controls': 87, 'Interior-Wall-Sconce': 88, 'Isolation-Module-FA': 89, 'Keypad': 90,
            "Linear-Strip-Light-4'-Ceiling-Mount": 91, "Linear-Strip-Light-4'-Chain-Hung": 92,
            "Linear-Strip-Light-4'-Wall-Mount": 93, 'Linear-Strip-Light-Ceiling-Mount': 94,
            'Linear-Strip-Light-Wall-Mount': 95, 'Low-Voltage-Relay-Panel': 96, 'Microphone-Audio-Outlet': 97,
            'Mini-Horn-FA-Piezo-Wall-Mount': 98, 'Monitoring-Module-FA': 99, 'Motion-Sensor-Security': 100,
            'Motion-Sensor-Security-Ceiling-Mount': 101, 'Motion-Sensor-Security-Wall-Mount': 102,
            'Occupancy-Sensor-Ceiling-Mount': 103, 'Occupancy-Sensor-Switch': 104, 'Occupancy-Sensor-Wall-Mount': 105,
            'PA-Speaker-Ceiling-Mount': 106, 'PA-Speaker-Wall-Mount': 107, 'Pac-Pole-Data': 108, 'Pac-Pole-Power': 109,
            'Pac-Pole-Power-Data': 110, 'Panelboard-Flush-Mount': 111, 'Pendant-Fixture': 112,
            'Pole-Mount-Luminaire': 113, 'Power-Data-Combination-Floor-Box': 114, 'Pull-Station--FA-Addressable': 115,
            'Push-Button-Egress': 116, 'Quad-Receptacle-Floor-Mount-15A': 117,
            'Recessed-Downlight-Emergency-Circuit': 118, 'Recessed-Downlight-Round': 119,
            'Recessed-Downlight-Square': 120, 'Recessed-Fluorescent-Luminaire': 121, 'Recessed-LED-Luminaire': 122,
            'Recessed-LED-Luminaire-Emergency-Circuit': 123, 'Recessed-Luminaire': 124,
            'Recessed-Luminaire-Emergency-Circuit': 125, 'Relay-Module-FA': 126, 'Remote-Annunciator-FA': 127,
            'Remote-DC-2-Heads': 128, 'Request-To-Exit': 129, 'Security-Intrusion-Horn': 130, 'Service-Ground-Bar': 131,
            'Smoke-CO-Detector': 132, 'Smoke-Detector-120V': 133, 'Smoke-Detector-Addressable': 134,
            'Solenoid-Actuator-Addressable': 135, 'Speaker-Emergency-Alarm': 136, 'Speaker-Sound-System': 137,
            'Special-Receptacle': 138, 'Special-Receptacle-Ceiling-Mount': 139, 'Sprinkler-Flow-Addressable': 140,
            'Sprinkler-Tamper-Addressable': 141, 'Starter-Hand-Off-Auto-Selector-Switch': 142,
            'Starter-Magnetic-Forward-Reversing': 143, 'Starter-Magnetic-Full-Voltage': 144,
            'Starter-Magnetic-Non-Reversing': 145, 'Strobe-FA-Ceiling-Mount': 146, 'Strobe-FA-Wall-Mount': 147,
            'Surface--Ceiling-Fluorescent-Luminaire': 148, 'Surface-Ceiling-LED-Luminaire': 149,
            'Surface-Mount-Downlight': 150, 'Surface-Mount-Downlight-Emergency-Circuit': 151,
            'Surface-Mount-Fluorescent-Luminaire-Emergency-Circuit': 152,
            'Surface-Mount-LED-Luminaire-Emergency-Circuit': 153, 'Surface-Mounted-Luminaire-EM-Power': 154,
            'Switch-3-Way': 155, 'Switch-4-Way': 156, 'Switch-Bank': 157, 'Switch-Bank-Low-Voltage-Dimmable': 158,
            'Switch-Dimmer-0-10V': 159, 'Switch-Key': 160, 'Switch-Low-Voltage': 161, 'Switch-Scene-Controller': 162,
            'Switch-Speed': 163, 'Switch-Timer': 164, 'Switch-With-Pilot-Light': 165,
            'Television-Outlet-Counter-Height': 166, 'Television-Outlet-Floor-Mount': 167,
            'Television-Outlet-Wall-Mount': 168, 'Thermostat-Mech-Controls': 169,
            'Three-Button-Controller-Up-Down-Stop': 170, 'Time-Clock-Astronomical': 171, 'Time-Clock-Mechanical': 172,
            'Track-Light-Length-Heads-TBD': 173, 'Vacancy-Sensor-Ceiling-Mount': 174,
            'Voice-Outlet-Counter-Height': 175, 'Voice-Outlet-Floor-Mount': 176, 'Wall-Junction-Box': 177,
            'Wall-Mount-Downlight': 178, 'Wall-Mount-Downlight-Emergency-Circuit': 179,
            'Wall-Mount-Fluorescent-Luminaire': 180, 'Wall-Mount-Fluorescent-Luminaire-Emergency-Circuit': 181,
            'Wall-Mount-LED-Luminaire': 182, 'Wall-Mount-LED-Luminaire-Emergency-Circuit': 183,
            'Wall-Mount-Luminaire-Emergency-Circuit': 184, 'Wall-Washer-Luminaire': 185, 'Wireless-Access-Point': 186,
            'Data-Outlet': 187, 'Voice-Outlet': 188}

inv_map = {v: k for k, v in mappings.items()}
classes = list(mappings.keys())
data_list = next(os.walk('data_symbol_templates'))[1]
data_dict = [{i: 0} for i in range(len(data_list))]
dataset_path = "G:\\dataset6"
dataset_label_path = os.path.join(dataset_path, "labels")
dataset_train_label_path = os.path.join(dataset_label_path, "train")
dataset_val_label_path = os.path.join(dataset_label_path, "val")

