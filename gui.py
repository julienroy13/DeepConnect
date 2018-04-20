import numpy as np
import time
import sys
import os
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg 
import utils
from agent import smart, MLP, random
import argparse

from env import Connect4, InvalidMove, Connect4Environment


# For some reason, this makes sure the icon can be displayed properly in Windows taskbar
import ctypes
import platform
if platform.system() == "Windows":
    myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

# GUI MANAGER
class GuiManager(QtWidgets.QWidget):

    def __init__(self, env, player2=None, player2_type='human', who_starts='flip_coin'):
        super(GuiManager, self).__init__() # Initializes the base class QMainWindow

        self.env = env
        self.player2 = player2
        self.player2_type = player2_type
        self.who_starts = who_starts

        # Initializes the GUI widgets and layout
        self.setupGUI()
        self.reset()

    
    # WIDGETS AND LAYOUT ----------------------------------------------------------------------------------------------
    def setupGUI(self):

        # Sets the name, incon and size of the main window
        self.setWindowTitle('Connect4') 
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.resize(800,800)

        # Creates the buttons' size policy
        sizePolicyBtn = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Minimum)
        sizePolicyBtn.setHorizontalStretch(0)
        sizePolicyBtn.setVerticalStretch(0)

        # Reset button
        self.ResetBtn = QtGui.QPushButton("Reset")
        sizePolicyBtn.setHeightForWidth(self.ResetBtn.sizePolicy().hasHeightForWidth())
        self.ResetBtn.setSizePolicy(sizePolicyBtn)
        self.ResetBtn.setStyleSheet("background-color: gray; color: black; font-size: 20px; font: bold")
        self.ResetBtn.clicked.connect(self.reset)
        self.ResetBtn.setEnabled(True)

        # GraphicView
        self.Image = pg.ImageItem(autoLevels=False)
        self.Image.mousePressEvent = self.human_add_disk
        self.Plot = pg.PlotWidget(title="Connect4")
        self.Plot.hideAxis("bottom")
        self.Plot.hideAxis("left")
        self.Plot.addItem(self.Image)
        self.coeff = 21 # For pad_grid(). has to be odd.

        # Build lookup table
        lut = np.zeros((256,3), dtype=np.ubyte)
        lut[0,:] = 50
        lut[100,:] = [255, 0, 0]
        lut[200:,:] = [0, 0, 255]
        lut[255,:] = 255
        self.Image.setLookupTable(lut, update=True)


        # Instanciates the layout
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.Plot, 0, 0, 100, 100)
        self.layout.addWidget(self.ResetBtn, 100, 55, 2, 45)

        # Splash icon
        splash_pix = QtGui.QPixmap('icon.png')
        splash = QtGui.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
        splash.show()
        time.sleep(0)

        # Display the interface on the screen
        self.show()
        self.updatePlot()

    def reset(self):
        self.env.reset()
        self.updatePlot()

        # Decide which player starts the game
        if self.who_starts == 'flip_coin':
            self.env.game.turn = np.random.choice([1, 2])
        else:
            self.env.game.turn = int(self.who_starts)

        # The bot plays if it starts the game
        if self.env.game.turn == 2 and self.player2_type != 'human':
            self.bot_add_disk()
    
    def human_add_disk(self, event):
        if not(self.env.game.over):
            # Tries to add a disk (if the column isn't full)
            try:
                row = self.getPos(event)
                self.env.game.make_move(self.env.game.turn, row)
                self.env.game.check_win(1)
                self.env.game.check_win(2)
                self.updatePlot()
                
                if self.player2_type != 'human':
                    # Fake thinking delay before playing
                    QtCore.QTimer.singleShot(np.random.randint(100, 500), lambda: self.bot_add_disk())

            except InvalidMove as e:
                print(e)

    def bot_add_disk(self):
        if not(self.env.game.over):
            player2_action = self.player2.select_action()
            self.env.game.make_move(2, player2_action)
            self.env.game.check_win(2)
            self.updatePlot()

    def getPos(self , event):
        row = int(event.pos().x())
        row = int(row / self.coeff)
        
        return row

    def updatePlot(self):
        # Updates the plot with the current state of the game
        pretty_grid = utils.pad_grid(self.env.game.grid, self.env.game.win_indices, self.coeff)
        self.Image.setImage(image=np.transpose(np.flip(pretty_grid, axis=0)))
        self.Image.setLevels([0, 255], update=True)
        return





# If we run the file directly
if __name__ == '__main__':
    
    # ARGPARSE ------
    parser = argparse.ArgumentParser()
    parser.add_argument('--player2_type', type=str, default='human',
                        help='Type of player you are playing against') # 'human', 'random' or 'XXXX.pkl'
    parser.add_argument('--who_starts', type=str, default='flip_coin',
                        help='Which player starts the game (you are player 1)',
                        choices=['flip_coin', '1', '2'])
    args = parser.parse_args()


    # GAME RELATED STUFF ------

    # Instanciates the environment
    env = Connect4Environment()
    params = {"epsilon": 0., "gamma": 1., "lambda": 0.9, "alpha": 1e-3}

    # Loads the player2
    print(args.player2_type)
    print(args.player2_type.endswith('.pkl'))
    if args.player2_type == 'human':
        player2 = None
    elif args.player2_type == 'random':
        player2 = random(model=None, params=params, env=env, p=2) 
    elif args.player2_type.endswith('.pkl'):
        estimator = MLP(env.d*env.game.n_rows*env.game.n_columns, [160], 3, "sigmoid", "glorot", verbose=True)
        player2 = smart(model=estimator, params=params, env=env, p=2)
        player2.load(os.path.join('models', args.player2_type))
        
    else:
        raise ValueError('Unrecognized player type entered as input.')


    # LAUNCHES THE GUI -------
    global app
    app = QtWidgets.QApplication(sys.argv)                  # Every PyQt application must create an application object
    gui = GuiManager(env, player2, args.player2_type, args.who_starts)      # Create an object "GuiManager"
    sys.exit(app.exec_())                                   # Enter the main loop of the application