import numpy as np
import bokeh
from bokeh.palettes import Sunset8, Viridis8
from bokeh.plotting import figure, show, output_notebook, curdoc
from bokeh.models import PointDrawTool, FreehandDrawTool, ColumnDataSource, PointDrawTool
from bokeh.models.callbacks import CustomJS

def F(x,y):
    return 1.3*np.exp(-2.5*((x-1.3)**2 + (y-0.8)**2)) - 1.2*np.exp(-2*((x-1.8)**2 + (y-1.3)**2))
def DF(x,y,epsilon):
    DFx=(F(x+epsilon,y)-F(x,y))/epsilon
    DFy=(F(x,y+epsilon)-F(x,y))/epsilon
    return [DFx,DFy]

def pth(u,v,epsilon,nu=.1):
    xs,ys=[u],[v]
    x0,y0=u,v
    while True:
        grad = DF(x0,y0,epsilon)
        z=F(x0,y0)
        x0=x0-nu*grad[0]
        y0=y0-nu*grad[1]
        xs.append(x0)
        ys.append(y0)
        if np.abs(F(x0,y0)-z)<.0001:
            break
    return xs,ys

def handler(attr,old,new):
    x,y=new['x'][-1],new['y'][-1]
    xs,ys = pth(x,y,.01,.1)
    path.data={'x':xs,'y':ys}


# Data to contour is the sum of two Gaussian functions.
x, y = np.meshgrid(np.linspace(0, 3, 100), np.linspace(0, 2, 100))
z = 1.3*np.exp(-2.5*((x-1.3)**2 + (y-0.8)**2)) - 1.2*np.exp(-2*((x-1.8)**2 + (y-1.3)**2))
sketch=ColumnDataSource({'x':[],'y':[]})
path = ColumnDataSource({'x':[],'y':[]})
p = figure(width=800, height=600, x_range=(0, 3), y_range=(0, 2),tooltips=[("(x,y)","($x,$y)")])
levels = np.linspace(-1, 1, 9)
contour_renderer = p.contour(x, y, z, levels, fill_color=Viridis8, line_color="black",fill_alpha=.3)
colorbar = contour_renderer.construct_color_bar()
p.add_layout(colorbar, "right")
l=p.circle(x='x',y='y',source=sketch)
p.line(x='x',y='y',source=path)
tool=PointDrawTool(renderers=[l])
p.add_tools(tool)
sketch.on_change('data',handler)


curdoc().add_root(p)