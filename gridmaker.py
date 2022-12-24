# Import relevant modules
import keras.models
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import tkinter as tk
import tkinter.font as font

from main import predict_model


class DrawableGrid(tk.Frame):
    def __init__(self, parent, width, height, size=5):
        super().__init__(parent, bd=1, relief="sunken")
        self.width = width
        self.height = height
        self.size = size
        self.pixels = None
        canvas_width = width * size
        canvas_height = height * size
        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0, width=canvas_width, height=canvas_height)
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)

        for row in range(self.height):
            for column in range(self.width):
                x0, y0 = (column * size), (row * size)
                x1, y1 = (x0 + size), (y0 + size)
                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             fill="white", outline="gray",
                                             tags=(self._tag(row, column), "cell"))
        self.canvas.tag_bind("cell", "<B1-Motion>", self.paint)
        self.canvas.tag_bind("cell", "<1>", self.paint)
        self.canvas.tag_bind("cell", "<B3-Motion>", self.erase)
        self.canvas.tag_bind("cell", "<3>", self.erase)

    def _tag(self, row, column):
        """Return the tag for a given row and column"""
        tag = f"{row},{column}"
        return tag

    def get_pixels(self):
        row = ""
        output = []
        for row in range(self.height):
            row_array = []
            for column in range(self.width):
                color = self.canvas.itemcget(self._tag(row, column), "fill")
                value = 1 if color == "black" else 0
                row_array.append(value)
            output.append(row_array)
        #  Reshape input to fit into the model
        output = np.array(output)
        self.pixels = np.array(output)
        self.pixels = self.pixels.reshape(28, 28)
        self.pixels = np.expand_dims(self.pixels, axis=-1)
        self.pixels = np.expand_dims(self.pixels, axis=0)

    def reset(self):
        cell = self.canvas.find_all()
        for i in cell:
            self.canvas.itemconfigure(i, fill="white")

    def paint(self, event):
        cell = self.canvas.find_closest(event.x, event.y)
        self.canvas.itemconfigure(cell, fill="black")
        if cell[0] % 28 != 0 and cell[0] < 784:
            self.canvas.itemconfigure((cell[0] + 1,), fill="black")  # Makes cell to the right
        if cell[0] % 28 != 1:
            self.canvas.itemconfigure((cell[0] - 1,), fill="black")  # Makes cell to the right
        if cell[0] < 757:
            self.canvas.itemconfigure((cell[0] + 28,), fill="black")  # Makes cell to the bottom
        if cell[0] > 28:
            self.canvas.itemconfigure((cell[0] - 28,), fill="black")  # Makes cell to the top

    def erase(self, event):
        cell = self.canvas.find_closest(event.x, event.y)
        self.canvas.itemconfigure(cell, fill="white")
        if cell[0] % 28 != 0 and cell[0] < 784:
            self.canvas.itemconfigure((cell[0] + 1,), fill="white")  # Makes cell to the right
        if cell[0] % 28 != 1:
            self.canvas.itemconfigure((cell[0] - 1,), fill="white")  # Makes cell to the right
        if cell[0] < 757:
            self.canvas.itemconfigure((cell[0] + 28,), fill="white")  # Makes cell to the bottom
        if cell[0] > 28:
            self.canvas.itemconfigure((cell[0] - 28,), fill="white")  # Makes cell to the top

    def display_number(self, model):
        self.get_pixels()
        prediction = predict_model(model, self.pixels[:1])
        # prediction = 10
        answer.configure(text=f'Answer:\n{prediction}')


global answer


def make_grid(model):
    root = tk.Tk()
    # my_model = model

    canvas = DrawableGrid(root, width=28, height=28, size=6)
    # b = tk.Button(root, text="Print Data", command=canvas.get_pixels)
    r = tk.Button(root, text="Reset", command=canvas.reset)
    global answer
    answer = tk.Button(root, text=f"Answer:\n", height=5, width=7, command=lambda: canvas.display_number(model))
    my_font = font.Font(size=10, weight="bold")
    answer['font'] = my_font
    # b.pack(side="top")
    r.pack(side="top")
    answer.pack(side="right")
    canvas.pack(fill="both", expand=True)
    root.mainloop()
