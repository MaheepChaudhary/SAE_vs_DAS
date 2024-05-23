from config import *
from imports import *

def visualization(loss, acc, title):

    # Create figure
    epochs = list(range(len(acc)))
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=epochs, y=acc, 
                             mode='lines+markers', 
                             name='Accuracy', 
                             yaxis='y1', 
                             marker=dict(color='blue')))
    
    fig.add_trace(go.Scatter(x=epochs, y=loss, 
                             mode='lines+markers', 
                             name='Loss', 
                             yaxis='y2', 
                             marker=dict(color='red')))

    # Create axis objects
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis=dict(
            title="Accuracy",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue")
        ),
        yaxis2=dict(
            title="Loss",
            titlefont=dict(color="red"),
            tickfont=dict(color="red"),
            anchor="free",
            overlaying="y",
            side="right",
            position=1
        )
    )

    # Update layout
    fig.update_layout(
        autosize=False,
        width=700,
        height=500,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        )
    )

    # Show plot
    # fig.show()
    fig.write_image(f"stats/fig_{name}.png")


def visualization_each_class(arr, title):

    scatter = go.Scatter(x=[i for i in range(len(arr))], 
                         y= arr, 
                         mode='markers', 
                         name='Data Points',
                         marker=dict(size=12))

    # Create a line plot
    line = go.Scatter(x=[i for i in range(len(arr))], 
                      y=arr, 
                      mode='lines', 
                      name='Trend Line')

    # Combine them in one figure
    fig = go.Figure(data=[scatter, line])

    # Add title and labels
    fig.update_layout(title = title,
                        xaxis_title='Classes',
                        yaxis_title='Respective classes Accuracy')

    fig.write_image(f"stats/fig_{name}_each_class.png")