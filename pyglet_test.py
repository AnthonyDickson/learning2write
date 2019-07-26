import pyglet

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
    square_outline = pyglet.graphics.vertex_list(4, ('v2f', square_verts))


    @window.event
    def on_draw():
        window.clear()
        pyglet.gl.glColor3f(0, 0, 0)
        line.draw(pyglet.gl.GL_LINE_LOOP)
        square.draw(pyglet.gl.GL_TRIANGLES)

        pyglet.gl.glColor3f(1, 0, 0)
        # square_outline.draw(pyglet.gl.GL_LINE_LOOP)
        pyglet.graphics.draw(4, pyglet.gl.GL_LINE_LOOP, ('v2i', square_verts))


    pyglet.app.run()
