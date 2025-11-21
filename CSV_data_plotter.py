import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import io

st.set_page_config(page_title="Distribution Fitter", layout="wide")

DISTRIBUTIONS = {
    'Normal': stats.norm,
    'Gamma': stats.gamma,
    'Weibull': stats.weibull_min,
    'Exponential': stats.expon,
    'Log-Normal': stats.lognorm,
    'Beta': stats.beta,
    'Chi-Square': stats.chi2,
    'Rayleigh': stats.rayleigh,
    'Uniform': stats.uniform,
    'Cauchy': stats.cauchy,
    'Student-t': stats.t,
    'Pareto': stats.pareto,
}

def calculate_fit_quality(data, fitted_dist, params):
    hist, bin_edges = np.histogram(data, bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fitted_values = fitted_dist.pdf(bin_centers, *params)

    errors = np.abs(hist - fitted_values)
    avg_error = np.mean(errors)
    max_error = np.max(errors)

    return avg_error, max_error

st.title("Distribution Fitter")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Data Input")

    input_method = st.radio("Select input method:",
                            ["Manual Entry", "Upload CSV", "Random Dataset"])

    data = None

    if input_method == "Manual Entry":
        data_text = st.text_area("Enter data (comma or space separated):",
                                 height=150)
        if data_text:
            try:
                data = np.array([float(x) for x in data_text.replace(',', ' ').split()])
            except:
                st.error("Invalid data format")

    elif input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            column = st.selectbox("Select column:", df.columns)
            data = df[column].dropna().values

    else:
        dist_type = st.selectbox("Distribution type:",
                                ["Normal", "Gamma", "Exponential", "Uniform"])
        n_samples = st.slider("Number of samples:", 50, 1000, 200)

        if st.button("Generate"):
            if dist_type == "Normal":
                data = np.random.normal(100, 15, n_samples)
            elif dist_type == "Gamma":
                data = np.random.gamma(2, 2, n_samples)
            elif dist_type == "Exponential":
                data = np.random.exponential(2, n_samples)
            else:
                data = np.random.uniform(0, 10, n_samples)
            st.session_state['generated_data'] = data

        if 'generated_data' in st.session_state:
            data = st.session_state['generated_data']

    if data is not None and len(data) > 0:
        st.success(f"Loaded {len(data)} data points")

        st.subheader("Distribution")
        selected_dist = st.selectbox("Select distribution:", list(DISTRIBUTIONS.keys()))

        fitting_mode = st.radio("Fitting mode:", ["Auto Fit", "Manual Fit"])

with col2:
    if data is not None and len(data) > 0:
        dist = DISTRIBUTIONS[selected_dist]

        if fitting_mode == "Auto Fit":
            params = dist.fit(data)

            st.subheader("Fitted Parameters")
            param_names = dist.shapes.split(',') if dist.shapes else []
            param_names = [p.strip() for p in param_names]
            param_names += ['loc', 'scale']

            for name, value in zip(param_names, params):
                st.write(f"**{name}:** {value:.4f}")

            avg_error, max_error = calculate_fit_quality(data, dist, params)

            st.subheader("Fit Quality")
            st.write(f"**Average Error:** {avg_error:.6f}")
            st.write(f"**Maximum Error:** {max_error:.6f}")

        else:
            st.subheader("Manual Parameter Adjustment")

            num_params = len(dist.fit(data))
            param_values = []

            param_names = dist.shapes.split(',') if dist.shapes else []
            param_names = [p.strip() for p in param_names]
            param_names += ['loc', 'scale']

            fitted_params = dist.fit(data)

            for i, (name, fitted_val) in enumerate(zip(param_names, fitted_params)):
                if name == 'loc':
                    val = st.slider(f"{name}",
                                   float(np.min(data)),
                                   float(np.max(data)),
                                   float(fitted_val))
                elif name == 'scale':
                    val = st.slider(f"{name}",
                                   0.1,
                                   float(np.max(data) - np.min(data)),
                                   float(fitted_val))
                else:
                    val = st.slider(f"{name}",
                                   0.1,
                                   10.0,
                                   float(fitted_val))
                param_values.append(val)

            params = tuple(param_values)

            avg_error, max_error = calculate_fit_quality(data, dist, params)

            st.subheader("Fit Quality")
            st.write(f"**Average Error:** {avg_error:.6f}")
            st.write(f"**Maximum Error:** {max_error:.6f}")

        st.subheader("Visualization")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(data, bins=30, density=True, alpha=0.6, color='blue', label='Data')

        x_range = np.linspace(np.min(data), np.max(data), 200)
        ax.plot(x_range, dist.pdf(x_range, *params), 'r-', lw=2,
                label=f'Fitted {selected_dist}')

        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()
