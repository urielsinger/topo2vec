import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput
from bokeh.plotting import figure


class DashboardGui:
    start_location = None
    folium_zoom = 14

    def __init__(self):
        # Set up data
        self.N = 200
        x = np.linspace(0, 4 * np.pi, self.N)
        y = np.sin(x)
        self.source = ColumnDataSource(data=dict(x=x, y=y))

        # Set up plot
        self.plot = figure(plot_height=400, plot_width=400, title="my sine wave",
                      tools="crosshair,pan,reset,save,wheel_zoom",
                      x_range=[0, 4 * np.pi], y_range=[-2.5, 2.5])

        self.plot.line('x', 'y', source=self.source, line_width=3, line_alpha=0.6)

        # Set up widgets
        self.text = TextInput(title="title", value='my sine wave')
        self.offset = Slider(title="offset", value=0.0, start=-5.0, end=5.0, step=0.1)
        self.amplitude = Slider(title="amplitude", value=1.0, start=-5.0, end=5.0, step=0.1)
        self.phase = Slider(title="phase", value=0.0, start=0.0, end=2 * np.pi)
        self.freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1, step=0.1)

        self.text.on_change('value', self.update_title)

        for w in [self.offset, self.amplitude, self.phase, self.freq]:
            w.on_change('value', self.update_data)

        # Set up layouts and add to document
        inputs = column(self.text, self.offset, self.amplitude,
                        self.phase, self.freq)

        curdoc().add_root(row(inputs, self.plot, width=800))
        curdoc().title = "Sliders"

    # Set up callbacks
    def update_title(self, attrname, old, new):
        self.plot.title.text = self.text.value


    def update_data(self, attrname, old, new):
        # Get the current slider values
        a = self.amplitude.value
        b = self.offset.value
        w = self.phase.value
        k = self.freq.value

        # Generate the new curve
        x = np.linspace(0, 4 * np.pi, self.N)
        y = a * np.sin(k * x + w) + b

        self.source.data = dict(x=x, y=y)


DashboardGui()


