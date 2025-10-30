import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "vscode"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

### Def functions 

def project_on_plane_xyz(df_sub: pd.DataFrame,
                         x="x", y="y", z="z") -> pd.DataFrame:
    """
    Orthogonally project every point (x,y,z) onto the plane z = x + y.
    """
    n_vec = np.array([-1., -1., 1.])          
    n_norm2 = n_vec.dot(n_vec)                

    pts = df_sub[[x, y, z]].values            
    factors = (pts @ n_vec) / n_norm2       
    proj = pts - factors[:, None] * n_vec    

    out = df_sub.copy()
    out[[x, y, z]] = proj
    return out


def plot_with_reconciliation(df: pd.DataFrame,
                            h: int,
                            Y_df : pd.DataFrame,
                            time_col : str = "time",
                            x_col : str = "x", 
                            y_col : str = "y", 
                            z_col : str = "z",
                            title: str = "Forecast reconciliation",
                            show_history : bool = False,
                             ):
    
    hist = df.iloc[-h:]
    fcst_raw = df.iloc[-h:]
    trues  = Y_df.pivot(columns = 'unique_id', values='y', index = 'ds')[-h:].rename(columns={'x1':'x','x2':'y','x3':"z"})
    trues['time'] = trues.index

    fcst_bu = fcst_raw.copy()
    fcst_bu[z_col] = fcst_bu[x_col] + fcst_bu[y_col]

    fcst_proj = project_on_plane_xyz(fcst_raw, x_col, y_col, z_col)

    xmin, xmax = df[x_col].min(), df[x_col].max()
    ymin, ymax = df[y_col].min(), df[y_col].max()
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 25),
                         np.linspace(ymin, ymax, 25))
    zz = xx + yy
    plane = go.Surface(x=xx, y=yy, z=zz,
                       opacity=.3, showscale=False,
                       colorscale="Greys", name="plane  z = x + y")

    def mk_trace(sub, name, color, dash, width=4, size=4, markers = True, use_cmap = False, cmap : str = 'Blues'):
        t = np.arange(len(sub[x_col].values))
        t_min, t_max = t.min(), t.max()
        if markers:
            mode = 'lines+markers'
        else:
            mode = 'lines'
        if use_cmap:               
            line_dict = dict(color=t,
                         colorscale=cmap,
                         cmin=-0.5*t_max,
                         cmax=t_max,
                         showscale=False,     
                         width=width, dash=dash)
        else:                     
            line_dict = dict(color=color, width=width, dash=dash)
        
        return go.Scatter3d(
            x=sub[x_col], y=sub[y_col], z=sub[z_col],
            mode=mode,
            marker=dict(size=size, color=color),
            line=line_dict,
            name=name,
            hovertemplate="t=%{customdata}<br>x=%{x:.2f}<br>"
                          "y=%{y:.2f}<br>z=%{z:.2f}",
            customdata=sub[time_col]
        )

   
    traces = [plane]
    if show_history :
        traces += [mk_trace(hist,      "history",    'blue', "solid", 3, 3, markers= False, use_cmap=True, cmap = 'Blues')]

    traces+= [
        mk_trace(fcst_raw,  "raw ARIMA",        "crimson",   "dash"),
        mk_trace(fcst_bu,   "bottom-up",        "seagreen",  "dot"),
        mk_trace(fcst_proj, "projection",       "orange",    "dashdot"),
        mk_trace(trues, "True vals",       "purple",    "solid", markers = False, use_cmap=True, cmap = 'Purples')
    ]


    for i in range(h):
        seg = go.Scatter3d(
            x=[fcst_raw.iloc[i][x_col],  fcst_proj.iloc[i][x_col]],
            y=[fcst_raw.iloc[i][y_col],  fcst_proj.iloc[i][y_col]],
            z=[fcst_raw.iloc[i][z_col],  fcst_proj.iloc[i][z_col]],
            mode="lines",
            line=dict(color="orange", width=1),
            showlegend=False, hoverinfo="skip")
        traces.append(seg)

    fig = go.Figure(traces)
    fig.update_layout(
        height=1000, width=1000, 
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="x3",
            xaxis=dict(showbackground=False,  
                   showgrid=False,       
                   zeroline=False),
            yaxis=dict(showbackground=False,
                    showgrid=False,
                    zeroline=False),
            zaxis=dict(showbackground=False,
                    showgrid=False,
                   zeroline=False),
            aspectmode="data"),
        legend=dict(bgcolor="rgba(255,255,255,.3)", x=0.02, y=0.98),
        paper_bgcolor = 'rgba(200,200,228,0.7)',
        # font_color = 'white'
    )

    return fig, fcst_bu, fcst_proj


def plot_3d_trajectory(x, y, z,
                       *,
                       title:str="",
                       color:str="mediumslateblue",
                       line_width:int=2,
                       show_slider:bool=True,
                       frame_step:int=50,
                       elev:int=30,
                       azim:int=45,
                       dist:int = 1,
                       fps:int = 24,
                       pad:int = 0.05,
                       colorscale:str = "Blues",
                       use_cmap : bool = True):
    
    t = np.arange(len(x))         
    t_min, t_max = t.min(), t.max()

   
    if use_cmap:               
        line_dict = dict(color=t,
                         colorscale=colorscale,
                         cmin=t_min-10,
                         cmax=t_max,
                         showscale=False,     
                         width=line_width)
    else:                     
        line_dict = dict(color=color, width=line_width)
    if show_slider:
        fig = go.Figure(
            go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]],
                         mode="lines",
                         line=line_dict)
        )
    else:
        fig = go.Figure(
            go.Scatter3d(x=x, y=y, z=z,
                         mode="lines",
                         line=line_dict)
        )
    if show_slider:
        frames = []
        for k in range(1, len(x), frame_step):
            frames.append(
                go.Frame(
                    name=str(k),
                    data=[go.Scatter3d(x=x[:k], y=y[:k], z=z[:k],
                                       mode="lines",
                                       line=line_dict)]
                )
            )
        fig.frames = frames

        slider_steps = [
            dict(method="animate",
                 args=[[f.name],
                       dict(mode="immediate",
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0))],
                 label=str(k))
            for k, f in enumerate(frames)
        ]
        span_x = np.ptp(x) or 1        
        span_y = np.ptp(y) or 1
        span_z = np.ptp(z) or 1

        xrange = [np.min(x) - pad * span_x, np.max(x) + pad * span_x]
        yrange = [np.min(y) - pad * span_y, np.max(y) + pad * span_y]
        zrange = [np.min(z) - pad * span_z, np.max(z) + pad * span_z]

        fig.update_layout(height=1000, width=1000, 
            legend=dict(bgcolor="rgba(255,255,255,.3)", x=0.02, y=0.98),
            paper_bgcolor = 'rgba(200,200,228,0.7)',
            sliders=[dict(active=0,
                          steps=slider_steps,
                          x=0.1, y=-0.07,
                          xanchor="left", yanchor="top",
                          len=0.8,
                          pad=dict(t=25))]
        )
        fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.05, x=0.0,          
            xanchor="left", yanchor="bottom",
            buttons=[
                dict(label="► Play",
                     method="animate",
                     args=[None,               
                           dict(frame=dict(duration=1000 / fps,
                                            redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate")]),
                dict(label="❚❚ Pause",
                     method="animate",
                     args=[[None],             
                           dict(frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0))])
            ])
        ]
    )
        
    fig.update_scenes(
        xaxis=dict(title="x", range=xrange, backgroundcolor = 'grey', gridcolor = 'black',zerolinecolor = 'black', autorange=False),
        yaxis=dict(title="y", range=yrange,backgroundcolor = 'grey', gridcolor = 'black',zerolinecolor = 'black',autorange=False),
        zaxis=dict(title="z", range=zrange,backgroundcolor = 'grey',gridcolor = 'black', zerolinecolor = 'black',autorange=False),
        aspectmode="manual",
        camera=dict(
            eye=dict(x=np.cos(np.deg2rad(azim)) * np.cos(np.deg2rad(elev)) * dist,
                     y=np.sin(np.deg2rad(azim)) * np.cos(np.deg2rad(elev)) * dist,
                     z=np.sin(np.deg2rad(elev)) * dist)
        )
    )
    fig.update_layout(title=title, showlegend=False)
    return fig




def gtop_P(W,S):
    Wi = np.linalg.inv(W)
    return S @ np.linalg.inv(S.T @ Wi @ S) @ S.T @ Wi




def plot_Pmats(P_mats,n, series):
    all_vals = np.concatenate([P.ravel() for P in P_mats.values()])
    vmax = all_vals.max()
    vmin = all_vals.min()
    abs_max = max(abs(vmin), abs(vmax))

    eps = 1e-2

    def pos(x):              
        return (x + abs_max) / (2*abs_max)

    cdict = [
        (0.00, 'navy'),                      
        (pos(-eps*5), 'steelblue'),           
        (pos(-eps),  'lightsteelblue'),      
        (pos(0.0),   'lightgrey'),            
        (pos(+eps),  'mistyrose'),            
        (pos(+eps*5),'salmon'),              
        (1.00, 'darkred')                    
    ]

    cmap = LinearSegmentedColormap.from_list('neg_zero_pos', cdict)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0., vmax=abs_max)
    sns.set(style='white', font_scale=0.8)
    n = len(P_mats)
    fig, axes = plt.subplots(1, n, figsize=(24, 19), squeeze=False)
    i = 0
    for ax, (title, P) in zip(axes[0], P_mats.items()):
        sns.heatmap(P,
                    cmap=cmap,
                    norm=norm,
                    square=True,
                    cbar= False,
                    xticklabels=series,
                    yticklabels=series,
                    ax=ax)
        i+=1
        ax.set_title(title)
        ax.set_xlabel('Base series j')
        ax.set_ylabel('Reconciled series i')
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm,
                        ax=axes.ravel().tolist(),
                        shrink=0.4,
                        pad=0.04)
    cbar.set_label('Element value  $P_{ij}$')
    plt.show()

if __name__ == "__main__":
    pass