import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff

def main():
    st.title("Bar Chart and 3D Graph Example")

    # Generate some sample data for the bar chart
    bar_data = pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D'],
        'Values': [10, 20, 15, 25]
    })

    # Display the data for the bar chart
    st.write("Sample Data for Bar Chart:")
    st.write(bar_data)

    # Create a bar chart
    st.write("Bar Chart:")
    fig_bar = go.Figure(data=[go.Bar(x=bar_data['Category'], y=bar_data['Values'])])
    st.plotly_chart(fig_bar)

    # Generate some sample data for the 3D surface plot
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    # Create a DataFrame for the 3D surface plot
    surface_data = pd.DataFrame({'X': X.flatten(), 'Y': Y.flatten(), 'Z': Z.flatten()})

    # Display the data for the 3D surface plot
    st.write("Sample Data for 3D Surface Plot:")
    st.write(surface_data.head())

    # Create a 3D surface plot
    st.write("3D Surface Plot:")
    fig_surface = go.Figure(data=[go.Surface(z=surface_data['Z'].values.reshape(100, 100),
                                             x=surface_data['X'].values.reshape(100, 100),
                                             y=surface_data['Y'].values.reshape(100, 100))])
    st.plotly_chart(fig_surface)

    # Generate some sample data for the confusion matrix
    confusion_matrix_data = np.array([[30, 10], [5, 55]])

    # Create a DataFrame for the confusion matrix
    cm_df = pd.DataFrame(confusion_matrix_data, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive'])

    # Display the data for the confusion matrix
    st.write("Confusion Matrix Data:")
    st.write(cm_df)

    # Create a confusion matrix graph
    st.write("Confusion Matrix:")
    fig_cm = ff.create_annotated_heatmap(z=confusion_matrix_data, x=['Predicted Negative', 'Predicted Positive'], y=['Actual Negative', 'Actual Positive'], colorscale='Viridis')
    fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted Label", yaxis_title="Actual Label")
    st.plotly_chart(fig_cm)

    # Generate some sample data for the heatmap
    np.random.seed(0)
    data = np.random.rand(10, 10)

    # Create a DataFrame for the heatmap
    heatmap_data = pd.DataFrame(data)

    # Display the data for the heatmap
    st.write("Sample Data for Heatmap:")
    st.write(heatmap_data.head())

    # Create a heatmap
    st.write("Heatmap:")
    fig_heatmap = go.Figure(data=go.Heatmap(z=data))
    st.plotly_chart(fig_heatmap)

    # Generate some sample data for the histogram
    np.random.seed(0)
    data = np.random.randn(1000)

    # Create a DataFrame for the histogram
    hist_data = pd.DataFrame({'Values': data})

    # Display the data for the histogram
    st.write("Sample Data for Histogram:")
    st.write(hist_data.head())

    # Create a histogram
    st.write("Histogram:")
    fig_hist = go.Figure(data=[go.Histogram(x=hist_data['Values'])])
    st.plotly_chart(fig_hist)

    # Generate some sample data for the scatter plot
    np.random.seed(0)
    x = np.random.randn(100)
    y = np.random.randn(100)

    # Create a DataFrame for the scatter plot
    scatter_data = pd.DataFrame({'X': x, 'Y': y})

    # Display the data for the scatter plot
    st.write("Sample Data for Scatter Plot:")
    st.write(scatter_data.head())

    # Create a scatter plot
    st.write("Scatter Plot:")
    fig_scatter = go.Figure(data=[go.Scatter(x=scatter_data['X'], y=scatter_data['Y'], mode='markers')])
    st.plotly_chart(fig_scatter)

if __name__ == "__main__":
    main()
