import sys
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

import colorama
colorama.init()
support = True

if colorama.Style.RESET_ALL == "":
    support = False

if support:
    print(f"[{bcolors.OKGREEN}INFO\t{bcolors.ENDC}] {bcolors.WARNING}hello", end=f'{bcolors.ENDC}')
else:
    print(f"[INFO\t] hello")




