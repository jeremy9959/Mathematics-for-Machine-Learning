import numpy as np
import bokeh
from bokeh.palettes import Sunset8, Viridis256
from bokeh.plotting import figure, show, output_file
from bokeh.models import PointDrawTool, FreehandDrawTool, ColumnDataSource, PointDrawTool, Title
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
    
code="""
    function f(x,y) {
        return -1.3*Math.exp(-2.5*((x-1.3)**2 + (y-0.8)**2)) -1.2*Math.exp(-2*((x-.5)**2 + (y-2)**2)) ; 
    };
    function Df(x,y) {
        let DFx=(f(x+.01,y)-f(x,y))/.01 ; 
        let DFy=(f(x,y+.01)-f(x,y))/.01 ;  
        return {'x':DFx,'y':DFy} ; 
    }
    function pth(u,v) {
        let xs = [u] ; 
        let ys = [v] ; 
        let x0 = u;
        let y0 = v ; 
        for (let i=0;;i++) {
            let grad = Df(x0,y0) ; 
            let z = f(x0,y0) ; 
            x0 = x0 - .1*grad.x ;
            y0 =y0 -  .1*grad.y ;
            xs.push(x0) ; 
            ys.push(y0) ;
            if (Math.abs(f(x0,y0)-z)<.0001) {
                break ; 
                }
            }
        return {'xs':xs,'ys':ys}
    }
    let x=source.data["x"].at(-1);
    let y=source.data["y"].at(-1);
    let p = pth(x,y);
    console.log(p);
    path.data["x"]=p.xs ; 
    path.data["y"]=p.ys;
    endpt.data["x"]=[p.xs.at(-1)] ; 
    endpt.data["y"]=[p.ys.at(-1)] ;
    console.log(endpt.data) ; 
    path.change.emit() ;
    endpt.change.emit() ; 
"""

sketch=ColumnDataSource({'x':[],'y':[]})
endpt = ColumnDataSource({'x':[],'y':[]}) 
path = ColumnDataSource({'x':[],'y':[]})

x, y = np.meshgrid(np.linspace(-1, 3, 100), np.linspace(0, 3, 100))
z = -1.3*np.exp(-2.5*((x-1.3)**2 + (y-0.8)**2)) - 1.2*np.exp(-2*((x-.5)**2 + (y-2)**2))


customjs = CustomJS(args=dict(source=sketch,path=path,endpt=endpt),code=code)

# Data to contour is the sum of two Gaussian functions.



p = figure(width=800, height=600,toolbar_location=None)
p.add_layout(Title(text="Click on a point to show gradient descent from that point"),'above')
#p.add_layout(Title(text="$$Contour\ plot\ of\ f(x,y)=-1.3\mathrm{exp}(-2.5((x-1.3)^2+(y-0.8)^2))-1.2\mathrm{exp}(-2((x-0.5)^2+(y-2)^2))$$"),'above')

levels = np.linspace(-1.5, 0,25)
contour_renderer = p.contour(x, y, z, levels, fill_color=Viridis256, line_color="black",fill_alpha=.5)


l=p.circle(x='x',y='y',source=sketch,alpha=0)
p.asterisk(x='x',y='y',source=sketch,size=5)
p.asterisk(x='x',y='y',source=endpt,size=5)
p.line(x='x',y='y',source=path,line_color='red',line_width=4)
tool=PointDrawTool(renderers=[l],num_objects=1)
p.add_tools(tool)
p.toolbar.active_tap=tool
sketch.js_on_change('data',customjs)



output_file("./test.html")
show(p)