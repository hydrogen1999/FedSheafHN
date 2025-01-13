# fl/main.py

from fl.manager import FLManager

def fl_main(args):
    fm = FLManager(args)
    fm.start()
