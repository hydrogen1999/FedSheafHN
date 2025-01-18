# fl/main.py

from fl.manager import FLManager

def fl_main(args):
    manager = FLManager(args)
    manager.start()
