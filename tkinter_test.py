"""
ZetCode Tkinter tutorial

This program draws three
rectangles filled with different
colours.

Author: Jan Bodnar
Last modified: April 2019
Website: www.zetcode.com
"""

from tkinter import Tk, Canvas, Frame, BOTH


class Example(Frame):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        self.master.title("Colours")
        self.pack(fill=BOTH, expand=1)

        canvas = Canvas(self)
        # canvas.create_rectangle(30, 10, 120, 80,
        #     outline="#fb0", fill="#fb0")
        # canvas.create_rectangle(150, 10, 240, 80,
        #     outline="#f50", fill="#f50")
        # canvas.create_rectangle(270, 10, 370, 80,
        #     outline="#05f", fill="#05f")
        canvas.pack(fill=BOTH, expand=1)

        self.draw_board(canvas)

    def draw_board(self, canvas):
        # draw checkerboard
        for r in range(7, -1, -1):
            for c in range(8):
                coords = (c * 30 + 4, r * 30 + 4, c * 30 + 30, r * 30 + 30)
                canvas.create_rectangle(coords, fill='black', disabledfill='',
                                        width=2, state='disabled')


def main():

    root = Tk()
    ex = Example()
    root.geometry("400x100+300+300")
    root.mainloop()


if __name__ == '__main__':
    main()