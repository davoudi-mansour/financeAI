import plotly
import os

i_component = 5
target = 'High'

fig = plotly.io.read_json(os.path.join('./plot', 'i_' + str(i_component) + "_" + target + '.json'))

fig.show()
