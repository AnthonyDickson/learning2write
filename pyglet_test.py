import pyglet
from pyglet.window import key, mouse

if __name__ == '__main__':
    width, height = 640, 480
    window = pyglet.window.Window(width=width, height=height)
    pyglet.gl.glClearColor(1, 1, 1, 1)

    line_verts = [120, 240, 520, 240]
    line = pyglet.graphics.vertex_list(2, ('v2f', line_verts))

    x, y = 20, 440
    square_verts = [20, 440,
                    20, 460,
                    40, 460,
                    40, 440]
    square = pyglet.graphics.vertex_list_indexed(4, [0, 1, 2, 0, 2, 3], ('v2f', square_verts))


    @window.event
    def on_draw():
        window.clear()
        pyglet.gl.glColor3f(0, 0, 0)
        line.draw(pyglet.gl.GL_LINE_LOOP)
        square.draw(pyglet.gl.GL_TRIANGLES)


    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.A:
            print('The "A" key was pressed.')
        elif symbol == key.LEFT:
            print('The left arrow key was pressed.')
        elif symbol == key.ENTER:
            print('The enter key was pressed.')
        else:
            print('A key was pressed')


    @window.event
    def on_mouse_press(x, y, button, modifiers):
        print('Mouse event location coords: (%d, %d)' % (x, y))

        if button == mouse.LEFT:
            print('The left mouse button was pressed.')
        elif button == mouse.RIGHT:
            print('The right mouse button was pressed.')
        else:
            print('A mouse button was pressed.')


    pyglet.app.run()
