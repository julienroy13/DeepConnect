import numpy as np
import time
import sys
import os
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg 
import utils

from env import Connect4, InvalidMove


# For some reason, this makes sure the icon can be displayed properly in Windows taskbar
import ctypes
import platform
if platform.system() == "Windows":
    myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

# GUI MANAGER
class GuiManager(QtWidgets.QWidget):

    def __init__(self):
        super(GuiManager, self).__init__() # Initializes the base class QMainWindow

        # Instanciates a GameOfLife object
        self.game = Connect4()

        # Initializes the GUI widgets and layout
        self.setupGUI()
    
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
        self.Image.mousePressEvent = self.add_piece
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
        self.game.reset()
        self.updatePlot()
    
    def add_piece(self, event):
        if not(self.game.over):
            # Tries to add a piece (if the column isn't full)
            try:
                row = self.getPos(event)
                self.game.make_move(self.game.turn, row)
                self.game.check_win(1)
                self.game.check_win(2)
                self.updatePlot()
            except InvalidMove as e:
                print(e)

    def getPos(self , event):
        row = int(event.pos().x())
        row = int(row / self.coeff)
        
        return row

    def updatePlot(self):
        # Updates the plot with the current state of the game
        pretty_grid = utils.pad_grid(self.game.grid, self.game.win_indices, self.coeff)
        self.Image.setImage(image=np.transpose(np.flip(pretty_grid, axis=0)))
        self.Image.setLevels([0, 255], update=True)
        return





# If we run the file directly
if __name__ == '__main__':
    global app
    app = QtWidgets.QApplication(sys.argv)  # Every PyQt application must create an application object
    gui = GuiManager()                      # Create an object "GuiManager"
    sys.exit(app.exec_())                   # Enter the main loop of the application