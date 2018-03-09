from tkinter import *
from PIL import Image, ImageTk
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


class Jet_Curry_Gui():
    def __init__(self, file):
        self.gui = Tk()
        self.gui.title('FITS Image')
        self.file = file

        # Frame to display the image
        self.masterFrame = Frame(self.gui, height=50, width=50)
        self.masterFrame.grid(row=0, column=0)

        # Interactive frame to choose streams and display values
        self.iFrame = Frame(self.gui, height=50, width=50)
        self.iFrame.grid(row=1, column=0)

        self.xStartVariable = IntVar()
        self.yStartVariable = IntVar()
        self.xEndVariable = IntVar()
        self.yEndVariable = IntVar()

        self.xStartLabel = Label(self.iFrame, text='X').grid(row=1, column=1)
        self.yStartLabel = Label(self.iFrame, text='Y').grid(row=1, column=2)
        self.pixelValueLabel = Label(
            self.iFrame, text='Value').grid(
            row=1, column=3)

        self.startPointLabel = Label(
            self.iFrame, text='Start stream: ').grid(
            row=2, column=0)
        self.xStartEntry = Entry(
            self.iFrame,
            textvariable=self.xStartVariable,
            width=5)
        self.xStartEntry.grid(row=2, column=1)
        self.yStartEntry = Entry(
            self.iFrame,
            textvariable=self.yStartVariable,
            width=5)
        self.yStartEntry.grid(row=2, column=2)
        self.startPixelValueLabel = Label(self.iFrame, text='')
        self.startPixelValueLabel.grid(row=2, column=3)

        self.endPointLabel = Label(
            self.iFrame, text='End stream: ').grid(
            row=3, column=0)
        self.xEndEntry = Entry(
            self.iFrame,
            textvariable=self.xEndVariable,
            width=5)
        self.xEndEntry.grid(row=3, column=1)
        self.yEndEntry = Entry(
            self.iFrame,
            textvariable=self.yEndVariable,
            width=5)
        self.yEndEntry.grid(row=3, column=2)
        self.endPixelValueLabel = Label(self.iFrame, text='')
        self.endPixelValueLabel.grid(row=3, column=3)

        self.runButton = Button(
            self.iFrame,
            text='Run',
            command=self.close_gui).grid(
            row=4,
            column=3)

        self.fits_data = fits.getdata(self.file)
        plt.imsave('dummy.jpg', self.fits_data)

        self.img = Image.open('dummy.jpg')
        self.photo = ImageTk.PhotoImage(self.img)
        self.panel = Label(self.masterFrame, image=self.photo)
        self.clicks = list(range(2))
        # Left mouse click for upstream (start)
        self.panel.bind('<Button-1>', self.get_start_point)
        # Right mouse click for downstream (end)
        self.panel.bind('<Button-2>', self.get_end_point)
        self.panel.grid(row=1, column=0)

        os.remove('dummy.jpg')

        self.gui.mainloop()

    def get_start_point(self, event):
        self.clicks[0:2] = event.x, event.y
        self.xStartEntry.delete(0, END)
        self.xStartEntry.insert(0, self.clicks[0])
        self.yStartEntry.delete(0, END)
        self.yStartEntry.insert(0, self.clicks[1])
        self.startPixelValueLabel.configure(
            text=self.fits_data[event.y - 1, event.x - 1])
        self.startPixelValueLabel.grid(row=2, column=3)

    def get_end_point(self, event):
        self.clicks[0:2] = event.x, event.y
        self.xEndEntry.delete(0, END)
        self.xEndEntry.insert(0, self.clicks[0])
        self.yEndEntry.delete(0, END)
        self.yEndEntry.insert(0, self.clicks[1])
        self.endPixelValueLabel.configure(
            text=self.fits_data[event.y - 1, event.x - 1])
        self.endPixelValueLabel.grid(row=3, column=3)

    def close_gui(self):
        if self.xStartVariable.get() != '' and self.yStartVariable.get(
        ) != '' and self.xEndVariable.get() != '' and self.yEndVariable.get() != '':
            self.gui.destroy()
        else:
            print('Missing an input!')


if __name__ == "__main__":
    my_gui = Jet_Curry_Gui()
