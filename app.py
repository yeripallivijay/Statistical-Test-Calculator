import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats


st.set_page_config(
    page_title="Statistical Test Calculator",
    
    layout="centered"
)


st.title(" Statistical Test Calculator & Visualizer")
st.markdown("""
This tool calculates statistical test results and **visualizes** them. 
It compares your score against **Critical Values** to show if the result is significant.
""")


st.sidebar.header("Select Test Type")
test_type = st.sidebar.radio(
    "Choose a test:",
    ("Z-Test (One Sample)", "T-Test (One Sample)", "Chi-Square Test (2x2)")
)


def plot_distribution(x, y, critical_values, stat_score, test_name, is_two_tailed=True):
    """
    Generic function to plot statistical distributions using Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
  
    ax.plot(x, y, label=f'{test_name} Distribution', color='#333333', lw=2)
    ax.fill_between(x, y, alpha=0.1, color='gray')

    
    if is_two_tailed:
        
        neg_crit, pos_crit = critical_values
        ax.fill_between(x, y, where=(x <= neg_crit), color='red', alpha=0.3, label='Rejection Region')
        ax.fill_between(x, y, where=(x >= pos_crit), color='red', alpha=0.3)
        ax.axvline(neg_crit, color='red', linestyle=':', label=f'Critical: {neg_crit:.2f}')
        ax.axvline(pos_crit, color='red', linestyle=':')
    else:
       
        pos_crit = critical_values
        ax.fill_between(x, y, where=(x >= pos_crit), color='red', alpha=0.3, label='Rejection Region')
        ax.axvline(pos_crit, color='red', linestyle=':', label=f'Critical: {pos_crit:.2f}')

   
    ax.axvline(stat_score, color='green', linestyle='--', linewidth=2.5, label=f'Your Score: {stat_score:.2f}')

    ax.set_title(f"Visualization: {test_name}", fontsize=14)
    ax.set_xlabel("Test Statistic Value")
    ax.set_ylabel("Probability Density")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return fig


def show_conclusion(stat_score, critical_value, p_value, alpha):
    col1, col2, col3 = st.columns(3)
    col1.metric("Calculated Score", f"{stat_score:.4f}")
    col2.metric(f"Critical Value", f"±{critical_value:.4f}")
    col3.metric("P-Value", f"{p_value:.4f}")

    is_significant = abs(stat_score) > critical_value
    
    if is_significant:
        st.error(f"**Result: Reject Null Hypothesis**")
        st.write(f"The score ({stat_score:.4f}) falls into the **Red Rejection Region**.")
    else:
        st.success(f"**Result: Fail to Reject Null Hypothesis**")
        st.write(f"The score ({stat_score:.4f}) falls into the **Safe Gray Region**.")



if test_type == "Z-Test (One Sample)":
    st.header("One-Sample Z-Test")
    st.info("Use when Population Variance is **Known** and n > 30.")

    with st.form("z_test_form"):
        col1, col2 = st.columns(2)
        with col1:
            mu = st.number_input("Population Mean (μ)", value=100.0)
            sigma = st.number_input("Population Std Dev (σ)", value=15.0)
        with col2:
            x_bar = st.number_input("Sample Mean (x̄)", value=105.0)
            n = st.number_input("Sample Size (n)", value=50, step=1)
        
        alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
        submitted = st.form_submit_button("Calculate & Plot")
        
        if submitted:
            
            std_error = sigma / math.sqrt(n)
            z_score = (x_bar - mu) / std_error
            z_critical = stats.norm.ppf(1 - alpha/2)
            p_value = stats.norm.sf(abs(z_score)) * 2
            
        
            show_conclusion(z_score, z_critical, p_value, alpha)
            
            
            st.subheader("Distribution Graph")
           
            limit = max(4, abs(z_score) + 1)
            x = np.linspace(-limit, limit, 1000)
            y = stats.norm.pdf(x)
            
            fig = plot_distribution(x, y, (-z_critical, z_critical), z_score, "Standard Normal (Z)")
            st.pyplot(fig)



elif test_type == "T-Test (One Sample)":
    st.header("One-Sample T-Test")
    st.info("Use when Population Variance is **Unknown** or n < 30.")

    with st.form("t_test_form"):
        col1, col2 = st.columns(2)
        with col1:
            mu = st.number_input("Population Mean (μ)", value=50.0)
            s = st.number_input("Sample Std Dev (s)", value=5.0)
        with col2:
            x_bar = st.number_input("Sample Mean (x̄)", value=48.0)
            n = st.number_input("Sample Size (n)", value=20, step=1)
            
        alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
        submitted = st.form_submit_button("Calculate & Plot")
        
        if submitted:
            
            df = n - 1
            std_error = s / math.sqrt(n)
            t_score = (x_bar - mu) / std_error
            t_critical = stats.t.ppf(1 - alpha/2, df)
            p_value = stats.t.sf(abs(t_score), df) * 2
            
          
            show_conclusion(t_score, t_critical, p_value, alpha)
            
     
            st.subheader("Distribution Graph")
            limit = max(4, abs(t_score) + 1)
            x = np.linspace(-limit, limit, 1000)
            y = stats.t.pdf(x, df)
            
            fig = plot_distribution(x, y, (-t_critical, t_critical), t_score, f"T-Distribution (df={df})")
            st.pyplot(fig)



elif test_type == "Chi-Square Test (2x2)":
    st.header("Chi-Square Test of Independence")
    
    with st.form("chi_test_form"):
        st.write("Enter Observed Frequencies (2x2 Table):")
        col1, col2 = st.columns(2)
        with col1:
            r1c1 = st.number_input("Group A - Outcome 1", value=10, step=1)
            r2c1 = st.number_input("Group B - Outcome 1", value=15, step=1)
        with col2:
            r1c2 = st.number_input("Group A - Outcome 2", value=20, step=1)
            r2c2 = st.number_input("Group B - Outcome 2", value=5, step=1)
            
        alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
        submitted = st.form_submit_button("Calculate & Plot")
        
        if submitted:
            observed = np.array([[r1c1, r1c2], [r2c1, r2c2]])
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)
            chi2_critical = stats.chi2.ppf(1 - alpha, dof)
            
           
            st.caption(f"Degrees of Freedom: {dof}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Chi2 Statistic", f"{chi2_stat:.4f}")
            col2.metric("Critical Value", f"{chi2_critical:.4f}")
            col3.metric("P-Value", f"{p_value:.4f}")
            
            if chi2_stat > chi2_critical:
                st.error("**Reject Null Hypothesis**")
            else:
                st.success("**Fail to Reject Null Hypothesis**")

          
            st.subheader("Distribution Graph")
           
            limit = max(10, chi2_stat + 5, chi2_critical + 5)
            x = np.linspace(0, limit, 1000)
            y = stats.chi2.pdf(x, dof)
            
            fig = plot_distribution(x, y, chi2_critical, chi2_stat, "Chi-Square Distribution", is_two_tailed=False)
            st.pyplot(fig)


st.markdown("---")
st.caption("Visualizer: Green Dashed Line = Your Result | Red Shaded Area = Critical Region")
