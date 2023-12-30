from scipy import interpolate
import plotly.graph_objects as go

def show_graph(X, array_acc, label='1', smooth=0, title='title', xaxis_label='x', yaxis_label='y'):
    tck = interpolate.splrep(X, array_acc, k = 3, s = smooth)
    accuracy_smoothed = interpolate.splev(X, tck, der = 0)

    data = [go.Scatter(x=X, y = accuracy_smoothed, name = label, mode='lines')]

    layout = go.Layout(
        height = 1000,
        title = dict(
            text = title,
            font_size = 30,
            x = .5
        ),
        xaxis_title = dict(
            text = xaxis_label,
            font_size = 20
        ),
        yaxis_title = dict(
            text = yaxis_label,
            font_size = 20
        ),
        legend = dict(
            x = 0.02, y = .98,
            bgcolor = 'rgba(0,0,0,0)',
            font_size = 20
        ),
        margin={'r':40}
    )
    return go.Figure(data, layout)

def show_graph_with_val(X, array_acc, array_val_acc, label_1='1', label_2='2', smooth_1=0, smooth_2=0, title='title', xaxis_label='x', yaxis_label='y'):
    tck = interpolate.splrep(X, array_acc, k = 3, s = smooth_1)
    accuracy_smoothed = interpolate.splev(X, tck, der = 0)

    tck = interpolate.splrep(X, array_val_acc, k = 3, s = smooth_2)
    val_accuracy_smoothed = interpolate.splev(X, tck, der = 0)

    data = [go.Scatter(x=X, y = accuracy_smoothed, name = label_1, mode='lines'),
            go.Scatter(x=X, y = val_accuracy_smoothed, name = label_2, mode='lines')]

    layout = go.Layout(
        height = 1000,
        title = dict(
            text = title,
            font_size = 30,
            x = .5
        ),
        xaxis_title = dict(
            text = xaxis_label,
            font_size = 20
        ),
        yaxis_title = dict(
            text = yaxis_label,
            font_size = 20
        ),
        legend = dict(
            x = 0.02, y = .98,
            bgcolor = 'rgba(0,0,0,0)',
            font_size = 20
        ),
        margin={'r':40}
    )
    return go.Figure(data, layout)

def show_bar(labels, accuracies, title='title', xaxis_label='x', yaxis_label='y'):
    data = [
        go.Bar(x = labels, y = accuracies*100),
    ]

    layout = go.Layout(
        height = 500,
        title = dict(
            text = title,
            font_size = 30,
            x = .5
        ),
        xaxis = dict(nticks = 11),
        xaxis_title = dict(
            text = xaxis_label,
            font_size = 20
        ),
        yaxis_range = [0,100],
        yaxis=dict(ticksuffix="%"),
        yaxis_title = dict(
            text = yaxis_label,
            font_size = 20
        ),
    )
    go.Figure(data, layout)
    return go.Figure(data, layout)