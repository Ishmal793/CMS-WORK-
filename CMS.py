import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime

# Function to load default data
@st.cache_data
def load_default_data():
    return pd.read_excel(
        'Modified_PPE_compliance_dataset.xlsx',
        sheet_name='Sheet1',
        engine='openpyxl'
    )

# Function to load uploaded files (supports Excel and CSV)
def load_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            st.sidebar.error("Unsupported file type! Please upload an Excel or CSV file.")
            st.stop()
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        st.stop()

# Sidebar for file upload or default dataset
st.sidebar.title("Upload or Load Dataset")

data_source = st.sidebar.radio(
    "Choose Data Source:",
    ("Default Dataset", "Upload Your Own Dataset")
)

# Load dataset based on user input
if data_source == "Default Dataset":
    data = load_default_data()
    st.sidebar.success("Default dataset loaded successfully!")
else:
    uploaded_file = st.sidebar.file_uploader("Upload an Excel or CSV file", type=['xlsx', 'csv'])

    if uploaded_file is not None:
        data = load_uploaded_file(uploaded_file)
        st.sidebar.success("Dataset uploaded successfully!")
    else:
        st.sidebar.warning("Please upload a dataset to proceed.")
        st.stop()

# Ensure 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date']).dt.date


# Define color palettes
default_colors = px.colors.qualitative.Plotly
time_series_colors = px.colors.qualitative.Set2

selected_analysis = st.sidebar.radio(
    "Select Analysis Level:",
    [
        "Variable Analytics",
        "Strategic Insights"
    ]
)


# Sidebar Filters

# Refresh Button
if st.button("Refresh Dashboard"):
    st.experimental_set_query_params()

# Tooltip Message
tooltip_message = (
    "The dataset is a working process. You cannot open the Excel file directly, "
    "and no modifications can be made. You can only add data to existing columns, "
    "and you cannot change the column names."
)
st.markdown(
    f'<span style="color: grey; font-size: 12px; text-decoration: underline;">{tooltip_message}</span>',
    unsafe_allow_html=True
)
analysis_variable=""
selected_insight=""
# Logic for Variable Analytics
if selected_analysis == "Variable Analytics":
    # Radio Button for Variable-Based Analytics
    analysis_variable = st.sidebar.radio(
        "Select Variable for Analysis:",
        [
            "Over All",
            "Analytics of Employee and Employees",
            "Analytics of Unit and Units",
            "Analytics of Shift and Shifts",
            "Analytics of Time Series",
            "Analytics of Camera Units"
        ]
    )



# Logic for Strategic Insights
elif selected_analysis == "Strategic Insights":
    # Radio Buttons for Selecting Insights
    selected_insight = st.sidebar.radio(
        "Select an Insight:",
        [
            "Combined Insights",
            "Critical Zone Insights",
            "Targets Monitoring Insights",
            "Time Tracking Insights",
            "Shift Productivity Insights",

            "Growth Tracker Insights",
            "Risk Radar Insights",


        ]
    )
# Data Filtering
st.sidebar.header("Filters")
analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Compliance", "Violation"])

min_date, max_date = min(data['Date']), max(data['Date'])
start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.sidebar.error("Start Date cannot be after End Date")
# Sidebar Filters
employee = st.sidebar.multiselect('Select Employee', options=data['Employee_Name'].unique())
shift = st.sidebar.multiselect('Select Shift', options=data['Shift'].unique())
factory = st.sidebar.multiselect('Select Factory', options=data['Factory'].unique())
department = st.sidebar.multiselect('Select Department', options=data['Department'].unique())
camera = st.sidebar.multiselect('Select Camera', options=data['Camera'].unique())
violation_type = st.sidebar.multiselect('Select Violation Type', options=data['Violation_Type'].unique())  # New filter for violation types

# Render charts based on the selected variable
filtered_data = data[
    (data['Date'] >= start_date) & (data['Date'] <= end_date) &
    (data['Employee_Name'].isin(employee) if employee else True) &
    (data['Shift'].isin(shift) if shift else True) &
    (data['Factory'].isin(factory) if factory else True) &
    (data['Department'].isin(department) if department else True) &
    (data['Camera'].isin(camera) if camera else True) &
    (data['Violation_Type'].isin(violation_type) if violation_type else True)  # Apply filter for violation types
]
# Determine the relevant data based on the analysis type
if analysis_type == "Violation":
    relevant_data = filtered_data[filtered_data['Violation_Type'] != 'Compliant']
    current_rate = (relevant_data.shape[0] / filtered_data.shape[0] * 100) if filtered_data.shape[0] > 0 else 0
    rate_label = "Current Violation Rate"
    relevant_checks = relevant_data.shape[0]
else:
    relevant_data = filtered_data[filtered_data['Violation_Type'] == 'Compliant']
    compliant_checks = relevant_data.shape[0]
    current_rate = (compliant_checks / filtered_data.shape[0] * 100) if filtered_data.shape[0] > 0 else 0
    rate_label = "Current Compliance Rate"
    relevant_checks = compliant_checks


print(f"{rate_label}: {current_rate:.2f}%")


 # Function to create pie charts
def create_pie_chart(data, group_by, title):
    pie_data = data[group_by].value_counts().reset_index()
    pie_data.columns = [group_by, 'Count']
    fig = px.pie(pie_data, names=group_by, values='Count', title=title)
    st.plotly_chart(fig, use_container_width=True)

# Function to create bar charts
def create_bar_chart(data, group_by, title, color_palette):
    grouped_data = data.groupby(group_by).size().reset_index(name='Count')
    grouped_data['Color'] = grouped_data.index.map(lambda x: color_palette[x % len(color_palette)])

    fig = px.bar(
        grouped_data, x=group_by, y='Count',
        title=title, color='Color',
        labels={'Count': 'Total Count'},
        color_discrete_sequence=color_palette
    )
    st.plotly_chart(fig, use_container_width=True)



    # Display the breakdown chart
    st.plotly_chart(fig_camera_breakdown, use_container_width=True)
if selected_analysis == "Variable Analytics":
    if analysis_variable == "Over All":
        # Select relevant data based on the analysis type
        total_checks = filtered_data.shape[0]

        # Display Header and Metrics
        st.header(f"Overall {analysis_type} Dashboard")

        # Create columns for metrics
        col1, col2, col3= st.columns(3)

        # Display Current Rate, Next Month Prediction, Total Checks, and Relevant Checks
        col1.metric(rate_label, f"{current_rate:.2f}%")

        col2.metric("Total Checks", total_checks)
        col3.metric("Relevant Checks", relevant_checks)
        # Group data for visualizations
        if analysis_type == "Violation":
            grouped_data = relevant_data.groupby(['Factory', 'Shift']).agg(
                Total_Violations=('Violation_Type', 'count')
            ).reset_index()
        else:
            grouped_data = relevant_data.groupby(['Factory', 'Shift']).agg(
                Total_Compliance=('Violation_Type', 'count')
            ).reset_index()

        # Factory-wise Violations/Compliance Gauge
        st.subheader(f"{analysis_type} by Factory")

        factory_colors = ['#00FF00', '#FF4500', '#1E90FF', '#FFFF00',
                          '#FF1493']  # Green, OrangeRed, DodgerBlue, Yellow, DeepPink (avoiding pink now)

        # Factory-wise Violations/Compliance Gauge
        col1, col2, col3 = st.columns(3)
        for i, (factory, count) in enumerate(grouped_data.groupby('Factory')[
                                                 f'Total_Violations' if analysis_type == "Violation" else 'Total_Compliance'].sum().items(),
                                             1):
            with [col1, col2, col3][i % 3]:
                color_index = i % len(factory_colors)  # Cycle through the color palette
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=count,
                    title={"text": f"Factory {factory} {analysis_type}"},
                    gauge={
                        'axis': {'range': [0, max(grouped_data.groupby('Factory')[
                                                      f'Total_Violations' if analysis_type == 'Violation' else 'Total_Compliance'].sum())]},
                        'bar': {'color': factory_colors[color_index]}
                    }
                ))

                st.plotly_chart(fig, use_container_width=True)

        # Shift-wise Violations/Compliance Gauge
        st.subheader(f"{analysis_type} by Shift")

        col4, col5 = st.columns(2)
        with col4:
            shift_value = grouped_data[grouped_data['Shift'] == 'Morning'][
                f'Total_Violations' if analysis_type == "Violation" else 'Total_Compliance'].sum()
            fig_morning = go.Figure(go.Indicator(
                mode="gauge+number",
                value=shift_value,
                title={"text": "Morning Shift"},
                gauge={
                    'axis': {'range': [0, max(grouped_data.groupby('Shift')[
                                                  f'Total_Violations' if analysis_type == 'Violation' else 'Total_Compliance'].sum())]},
                    'bar': {'color': '#32CD32'}  # LimeGreen color for Morning Shift
                }
            ))

            st.plotly_chart(fig_morning, use_container_width=True)

        with col5:
            shift_value = grouped_data[grouped_data['Shift'] == 'Evening'][
                f'Total_Violations' if analysis_type == "Violation" else 'Total_Compliance'].sum()
            fig_evening = go.Figure(go.Indicator(
                mode="gauge+number",
                value=shift_value,
                title={"text": "Evening Shift"},
                gauge={
                    'axis': {'range': [0, max(grouped_data.groupby('Shift')[
                                                  f'Total_Violations' if analysis_type == 'Violation' else 'Total_Compliance'].sum())]},
                    'bar': {'color': '#FF8C00'}  # DarkOrange for Evening Shift
                }
            ))

            st.plotly_chart(fig_evening, use_container_width=True)
        row_selection = st.radio("Choose Rows to Display:", ("First Five Rows", "Last Five Rows"))

        # Display data based on radio selection
        if row_selection == "First Five Rows":
            st.write("### First Five Rows of the Dataset")
            st.write(data.head())
        else:
            st.write("### Last Five Rows of the Dataset")
            st.write(data.tail())


    elif analysis_variable == "Analytics of Employee and Employees":
        st.header("Employee Analytics")

        if analysis_type == "Violation":
            relevant_data = filtered_data[filtered_data['Violation_Type'] != 'Compliant']
            title = "Employee Violations"
        else:
            relevant_data = filtered_data[filtered_data['Violation_Type'] == 'Compliant']
            title = "Employee Compliance"

        # Group data by Employee to count occurrences (first chart)
        employee_counts = relevant_data['Employee_Name'].value_counts().reset_index()
        employee_counts.columns = ['Employee_Name', 'Count']

        # Create a bar chart: Total Compliance/Violations by Employee
        fig_employee_compliance = px.bar(
            employee_counts,
            x='Employee_Name',
            y='Count',
            title=title,
            labels={'Count': 'Total Count'},
            color='Employee_Name',  # Assign distinct colors to each employee
            color_discrete_sequence=px.colors.qualitative.Plotly  # Use a qualitative palette
        )

        # Update layout of the first chart for better appearance
        fig_employee_compliance.update_layout(
            xaxis_title="Employee Name",
            yaxis_title="Count",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            font=dict(color="white")  # White font for contrast
        )

        # Second Chart: Breakdown by Violation/Compliance Type for Each Employee
        breakdown_data = relevant_data.groupby(['Employee_Name', 'Violation_Type']).size().reset_index(name='Count')

        fig_violation_breakdown = px.bar(
            breakdown_data,
            x='Employee_Name',
            y='Count',
            color='Violation_Type',  # Distinguish by Violation/Compliance Type
            title=f"{title} by Type",
            labels={'Count': 'Total Count', 'Violation_Type': 'Type'},
            barmode='group',  # Group bars for better comparison
            color_discrete_sequence=px.colors.qualitative.Set2  # Use a distinct color palette for types
        )

        # Update layout of the second chart for better appearance
        fig_violation_breakdown.update_layout(
            xaxis_title="Employee Name",
            yaxis_title="Count",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            font=dict(color="white")  # White font for good contrast
        )

        # Display both charts
        st.plotly_chart(fig_employee_compliance, use_container_width=True)
        st.plotly_chart(fig_violation_breakdown, use_container_width=True)


    elif analysis_variable == "Analytics of Unit and Units":
        st.header("Unit Analytics")
        st.header(f"{analysis_type} by Unit")

        # Define distinct colors for factories
        factory_colors = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf'  # Cyan
        ]

        # Factory-wise Violations/Compliance Chart
        st.subheader(f"{analysis_type} by Factory")

        factory_data = relevant_data.groupby(['Factory']).agg(
            Total_Count=('Violation_Type', 'count') if analysis_type == "Violation" else ('Employee_Name', 'count')
        ).reset_index()

        # Create a color map for factories
        factory_data['Color'] = factory_data.index.map(lambda x: factory_colors[x % len(factory_colors)])

        fig_factory = px.bar(factory_data, x='Factory', y='Total_Count',
                             title=f"{analysis_type} by Factory",
                             labels={
                                 'Total_Count': 'Total Violations' if analysis_type == "Violation" else 'Total Compliance'},
                             color='Color')  # Use the assigned color

        # Update layout for integer x-axis ticks
        fig_factory.update_layout(
            xaxis=dict(
                dtick=1,  # Set the tick interval to 1 for integer values
                tickmode='linear'  # Ensure ticks are linear
            )
        )

        st.plotly_chart(fig_factory, use_container_width=True)

        # Department-wise Violations/Compliance Chart
        st.subheader(f"{analysis_type} by Department")

        department_data = relevant_data.groupby(['Department']).agg(
            Total_Count=('Violation_Type', 'count') if analysis_type == "Violation" else ('Employee_Name', 'count')
        ).reset_index()

        # Define distinct colors for departments
        department_colors = [
            '#ffbb78',  # Light Orange
            '#98df8a',  # Light Green
            '#ff9896',  # Light Red
            '#c5b0d5',  # Light Purple
            '#f7b6d2',  # Light Pink
            '#c49c94',  # Light Brown
            '#f7f7f7',  # Light Gray
            '#dbdb8d',  # Light Olive
            '#9edae5',  # Light Cyan
            '#f3d9a4'  # Light Yellow
        ]

        # Create a color map for departments
        department_data['Color'] = department_data.index.map(lambda x: department_colors[x % len(department_colors)])

        fig_department = px.bar(department_data, x='Department', y='Total_Count',
                                title=f"{analysis_type} by Department",
                                labels={
                                    'Total_Count': 'Total Violations' if analysis_type == "Violation" else 'Total Compliance'},
                                color='Color')  # Use the assigned color

        # Update layout for integer x-axis ticks
        fig_department.update_layout(
            xaxis=dict(
                dtick=1,  # Set the tick interval to 1 for integer values
                tickmode='linear'  # Ensure ticks are linear
            )
        )

        st.plotly_chart(fig_department, use_container_width=True)
        # Visualization Logic for Violations
        if analysis_type == "Violation":
            # Filter for Violations
            violation_data = filtered_data[filtered_data['Violation_Type'] != 'Compliant']

            # Create a Bar Chart for Violations
            if not violation_data.empty:
                violation_count = violation_data['Violation_Type'].value_counts().reset_index()
                violation_count.columns = ['Violation Type', 'Count']

                # Define distinct colors for violation types using qualitative colors
                fig_violation = px.bar(violation_count, x='Violation Type', y='Count',
                                       title="Violation Counts",
                                       labels={'Count': 'Number of Violations'},
                                       color='Violation Type',  # Color by Violation Type for distinct colors
                                       color_discrete_sequence=px.colors.qualitative.Dark2)  # Darker color palette

                st.plotly_chart(fig_violation, use_container_width=True)
            else:
                st.write("No violation data available for the selected filters.")

    elif analysis_variable == "Analytics of Shift and Shifts":
        st.header("Shift Analytics")
        # Shift-wise Violations/Compliance Chart
        st.subheader(f"{analysis_type} by Shift")

        # Group data by shift to calculate total counts
        shift_data = relevant_data.groupby(['Shift']).agg(
            Total_Count=('Violation_Type', 'count') if analysis_type == "Violation" else ('Employee_Name', 'count')
        ).reset_index()

        # Bar chart: Overall Violations/Compliance by Shift
        fig_shift = px.bar(
            shift_data,
            x='Shift',
            y='Total_Count',
            title=f"{analysis_type} by Shift",
            labels={
                'Total_Count': 'Total Violations' if analysis_type == "Violation" else 'Total Compliance'
            },
            color='Shift',  # Assign distinct colors for each shift
            color_discrete_sequence=px.colors.qualitative.Plotly  # Use a vibrant color palette
        )

        # Update layout for better appearance
        fig_shift.update_layout(
            xaxis_title="Shift",
            yaxis_title="Total Count",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            font=dict(color="white")  # White font for contrast
        )

        # Display the shift-wise chart
        st.plotly_chart(fig_shift, use_container_width=True)

        # Shift-wise Breakdown by Type (Second Chart)
        st.subheader(f"{analysis_type} Breakdown by Shift and Type")

        # Group data by Shift and Violation/Compliance Type
        shift_breakdown = relevant_data.groupby(['Shift', 'Violation_Type']).size().reset_index(name='Count')

        # Bar chart: Breakdown of Types by Shift
        fig_shift_breakdown = px.bar(
            shift_breakdown,
            x='Shift',
            y='Count',
            color='Violation_Type',  # Color by type for distinction
            title=f"{analysis_type} Breakdown by Shift and Type",
            labels={'Count': 'Total Count', 'Violation_Type': 'Type'},
            barmode='group',  # Group bars for better comparison
            color_discrete_sequence=px.colors.qualitative.Set2  # Use another qualitative palette for better distinction
        )

        # Update layout for better appearance
        fig_shift_breakdown.update_layout(
            xaxis_title="Shift",
            yaxis_title="Count",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            font=dict(color="white")  # White font for good contrast
        )

        # Display the shift-wise breakdown chart
        st.plotly_chart(fig_shift_breakdown, use_container_width=True)


    elif analysis_variable == "Analytics of Time Series" :

        # Date-wise Violations/Compliance Chart (Line Chart)
        st.subheader(f"{analysis_type} by Date")

        # Group data by Date to calculate total counts
        date_data = relevant_data.groupby(['Date']).agg(
            Total_Count=('Violation_Type', 'count') if analysis_type == "Violation" else ('Employee_Name', 'count')
        ).reset_index()

        # Line chart: Overall Violations/Compliance Over Time
        fig_date = px.line(
            date_data,
            x='Date',
            y='Total_Count',
            title=f"{analysis_type} Over Time",
            labels={
                'Total_Count': 'Total Violations' if analysis_type == "Violation" else 'Total Compliance'
            },
            markers=True,  # Add markers for better visibility
            color_discrete_sequence=px.colors.qualitative.Set1  # Use Set1 color scheme
        )

        # Update layout for better appearance
        fig_date.update_layout(
            xaxis_title="Date",
            yaxis_title="Count",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            font=dict(color="white"),  # White font for contrast
        )

        # Display the line chart
        st.plotly_chart(fig_date, use_container_width=True)

        # Breakdown by Type Over Time (Area Chart)
        st.subheader(f"{analysis_type} Breakdown by Date and Type")

        # Group data by Date and Violation/Compliance Type
        date_breakdown = relevant_data.groupby(['Date', 'Violation_Type']).size().reset_index(name='Count')

        # Area chart: Breakdown by Violation/Compliance Type Over Time
        fig_date_breakdown = px.area(
            date_breakdown,
            x='Date',
            y='Count',
            color='Violation_Type',  # Distinguish by type
            title=f"{analysis_type} Breakdown Over Time",
            labels={'Count': 'Total Count', 'Violation_Type': 'Type'},
            color_discrete_sequence=px.colors.qualitative.Set2  # Use another qualitative palette
        )

        # Update layout for better appearance
        fig_date_breakdown.update_layout(
            xaxis_title="Date",
            yaxis_title="Count",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            font=dict(color="white")  # White font for contrast
        )

        # Display the area chart
        st.plotly_chart(fig_date_breakdown, use_container_width=True)



    elif analysis_variable == "Analytics of Camera Units":
        st.header("Camera Unit Analytics")

        # Camera-wise Violations/Compliance Chart
        st.subheader(f"{analysis_type} by Camera")

        # Grouping data by Camera to count occurrences of violations
        camera_data = relevant_data.groupby(['Camera']).agg(
            Total_Violations=(
                'Violation_Type', lambda x: (x != 'Compliant').sum() if analysis_type == "Violation" else len(x)
            )
        ).reset_index()

        # Create a bar chart for total violations by camera
        fig_camera = px.bar(
            camera_data,
            x='Camera',
            y='Total_Violations',
            title=f"Total {analysis_type} by Camera",
            labels={'Total_Violations': 'Number of Violations'},
            color='Total_Violations',  # Color by Total_Violations for distinct colors
            color_continuous_scale='YlOrRd'  # Yellow-Orange-Red color scale
        )

        # Update layout for better appearance
        fig_camera.update_layout(
            xaxis_title="Camera",
            yaxis_title="Number of Violations",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for better appearance
            font=dict(color="black"),  # Black font color for good contrast
            xaxis=dict(
                tickmode='array',
                tickvals=camera_data['Camera'],  # Set x-axis ticks to the camera names
                ticktext=camera_data['Camera']  # Display camera names as tick labels
            )
        )

        # Display the chart
        st.plotly_chart(fig_camera, use_container_width=True)

        # Breakdown by Type Over Camera (Second Chart)
        st.subheader(f"{analysis_type} Breakdown by Camera and Type")

        # Grouping data by Camera and Violation Type to count occurrences
        camera_breakdown = relevant_data.groupby(['Camera', 'Violation_Type']).size().reset_index(name='Count')

        # Create a grouped bar chart for breakdown by type for each camera
        fig_camera_breakdown = px.bar(
            camera_breakdown,
            x='Camera',
            y='Count',
            color='Violation_Type',  # Distinguish by type
            title=f"{analysis_type} Breakdown by Camera and Type",
            labels={'Count': 'Total Count', 'Violation_Type': 'Type'},
            barmode='group',  # Group bars for better comparison
            color_discrete_sequence=px.colors.qualitative.Set2  # Use another qualitative color palette
        )

        # Update layout for better appearance
        fig_camera_breakdown.update_layout(
            xaxis_title="Camera",
            yaxis_title="Count",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for better appearance
            font=dict(color="black"),  # Black font for good contrast
            xaxis=dict(
                tickmode='array',
                tickvals=camera_breakdown['Camera'].unique(),  # Set x-axis ticks to unique camera names
                ticktext=camera_breakdown['Camera'].unique()  # Display camera names as tick labels
            )
        )

        # Display the breakdown chart
        st.plotly_chart(fig_camera_breakdown, use_container_width=True)

elif selected_analysis == "Strategic Insights":
    # Combined Insights
    if selected_insight == "Combined Insights":
        st.subheader(f"{analysis_type} by Factory, Department")

        # Group data for Unit-wise Violations/Compliance
        grouped_unit_data = relevant_data.groupby(['Factory', 'Department']).agg(
            Total_Count=('Violation_Type', 'count') if analysis_type == "Violation" else ('Employee_Name', 'count')
        ).reset_index()

        # Add a numeric index for factories (1, 2, 3...)
        grouped_unit_data['Factory_Index'] = grouped_unit_data.groupby('Factory').ngroup() + 1

        # Color palette for the bar chart
        color_palette = px.colors.qualitative.Set3  # Vibrant color palette

        # Create a bar chart for Unit-wise Violations/Compliance
        fig_unit = px.bar(
            grouped_unit_data,
            x='Factory_Index',
            y='Total_Count',
            color='Department',
            title=f"{analysis_type} by Unit",
            labels={
                'Factory_Index': 'Factory ',
                'Total_Count': 'Total Violations' if analysis_type == "Violation" else 'Total Compliance'
            },
            color_discrete_sequence=color_palette  # Assign color palette
        )

        # Update layout for better appearance
        fig_unit.update_layout(
            xaxis_title="Factory ",
            yaxis_title="Count",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            font=dict(color="white"),  # White font color for dark background
            xaxis=dict(
                tickmode='linear',  # Ensure all numeric ticks are displayed
                tick0=1,  # Start at 1
                dtick=1  # Step size of 1
            )
        )

        # Display the bar chart
        st.plotly_chart(fig_unit, use_container_width=True)

        # Filter data based on user input
        if analysis_type == "Violation":
            relevant_data = data[data['Violation_Type'] != 'Compliant']
        else:
            relevant_data = data[data['Violation_Type'] == 'Compliant']


        # Function to create combined charts
        def combined_charts():
            fig = go.Figure()

            # Group data by Department and Shift
            department_shift_data = relevant_data.groupby(['Department', 'Shift']).agg(
                Total_Count=('Violation_Type', 'count')
            ).reset_index()

            # Add Department by Shift Bar Chart
            # Using distinct colors for each department
            department_colors = px.colors.qualitative.Pastel  # Color palette for department-wise chart

            for i, department in enumerate(department_shift_data['Department'].unique()):
                department_data = department_shift_data[department_shift_data['Department'] == department]
                color = department_colors[i % len(department_colors)]  # Cycle through the color palette
                fig.add_trace(go.Bar(
                    x=department_data['Shift'],
                    y=department_data['Total_Count'],
                    name=str(department),  # Ensure department name is used as legend entry
                    hoverinfo='text',
                    text=department_data['Total_Count'],
                    marker_color=color  # Set the color for each department
                ))

            # Update the layout with title and axis labels
            fig.update_layout(
                title=f"{analysis_type} by Department and Shift",
                barmode='stack',  # Stacked bar chart
                xaxis_title='Shift',
                yaxis_title='Total Count',
                legend_title='Department',
                template='plotly_white',  # White background for better visibility
                font=dict(color="white")  # White font for contrast
            )

            # Render the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)


        # Call the function to display combined charts
        combined_charts()
        # Factory by Trend Over Time
        trend_data = relevant_data.groupby(['Date', 'Factory']).agg(
            Total_Count=('Violation_Type', 'count')
        ).reset_index()

        # Create a new figure for the trend chart
        fig_trend = go.Figure()

        # Use a color palette for the Factory trends
        factory_colors = px.colors.qualitative.Vivid  # Vivid colors for factory trends

        for i, factory in enumerate(trend_data['Factory'].unique()):
            factory_data = trend_data[trend_data['Factory'] == factory]
            color = factory_colors[i % len(factory_colors)]  # Assign color to each factory
            fig_trend.add_trace(go.Scatter(
                x=factory_data['Date'],
                y=factory_data['Total_Count'],
                mode='lines+markers',
                name=str(factory),  # Ensure name is a string
                hoverinfo='text',
                text=factory_data['Total_Count'],
                line=dict(color=color, width=2),  # Set line color
                marker=dict(size=6, symbol='circle', color=color)  # Set marker color
            ))

        fig_trend.update_layout(
            title=f"{analysis_type} Trend Over Time by Factory",
            xaxis_title='Date',
            yaxis_title='Total Count',
            legend_title='Factory',
            template='plotly_white',  # Keep background white
            plot_bgcolor='rgba(0, 0, 0, 0)'  # Ensure the plot area background is transparent
        )

        st.plotly_chart(fig_trend, use_container_width=True)

        # Employee by Over Time
        employee_data = relevant_data.groupby(['Date', 'Employee_Name']).agg(
            Total_Count=('Violation_Type', 'count')
        ).reset_index()

        # Create a new figure for the employee trend chart
        fig_employee = go.Figure()

        # Use a color palette for Employee trends
        employee_colors = px.colors.qualitative.Set2  # Soft colors for employee trends

        for i, employee in enumerate(employee_data['Employee_Name'].unique()):
            emp_data = employee_data[employee_data['Employee_Name'] == employee]
            color = employee_colors[i % len(employee_colors)]  # Assign color to each employee
            fig_employee.add_trace(go.Scatter(
                x=emp_data['Date'],
                y=emp_data['Total_Count'],
                mode='lines+markers',
                name=str(employee),  # Ensure name is a string
                hoverinfo='text',
                text=emp_data['Total_Count'],
                line=dict(color=color, width=2),  # Set line color
                marker=dict(size=6, symbol='circle', color=color)  # Set marker color
            ))

        fig_employee.update_layout(
            title=f"{analysis_type} Over Time by Employee",
            xaxis_title='Date',
            yaxis_title='Total Count',
            legend_title='Employee',
            template='plotly_white',  # Keep background white
            plot_bgcolor='rgba(0, 0, 0, 0)'  # Ensure the plot area background is transparent
        )

        st.plotly_chart(fig_employee, use_container_width=True)

    # Critical Zone Insights
    elif selected_insight == "Critical Zone Insights":
        zone_colors = {'Green Zone': 'green', 'Yellow Zone': 'yellow', 'Red Zone': 'red'}

        if analysis_type == "Violation":
            st.subheader("Critical Zone Insights (Red: > 50% Violation Rate, Yellow: 30-50%, Green: < 30%)")

            # Calculate violation rates
            violation_rates = data.groupby(['Factory', 'Department'])['Violation_Type'].apply(
                lambda x: (x != 'Compliant').sum() / len(x) * 100
            ).reset_index(name='Violation Rate')

            # Assign zones
            violation_rates['Zone'] = pd.cut(
                violation_rates['Violation Rate'], bins=[0, 30, 50, 100],
                labels=['Green Zone', 'Yellow Zone', 'Red Zone']
            )

            # Display the table
            st.dataframe(violation_rates)

            # Factory-wise Violation Chart
            fig_factory_violations = px.bar(
                violation_rates,
                x=violation_rates.groupby('Factory').ngroup() + 1,  # Numeric x-axis
                y='Violation Rate',
                color='Zone',
                title="Critical Zone Violation Rates by Factory",
                labels={'x': 'Factory', 'Violation Rate': 'Violation Rate (%)'},
                color_discrete_map=zone_colors
            )

            fig_factory_violations.update_layout(
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                yaxis=dict(range=[0, 100]),
                plot_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(color="white")
            )
            st.plotly_chart(fig_factory_violations, use_container_width=True)

            # Department-wise Violation Chart
            fig_department_violations = px.bar(
                violation_rates,
                x=violation_rates.groupby('Department').ngroup() + 1,  # Numeric x-axis
                y='Violation Rate',
                color='Zone',
                title="Critical Zone Violation Rates by Department",
                labels={'x': 'Department', 'Violation Rate': 'Violation Rate (%)'},
                color_discrete_map=zone_colors
            )

            fig_department_violations.update_layout(
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                yaxis=dict(range=[0, 100]),
                plot_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(color="white")
            )
            st.plotly_chart(fig_department_violations, use_container_width=True)

            # Employees in the Red Zone
            red_zone_employees = data[
                (data['Violation_Type'] != 'Compliant') &
                (data.groupby(['Factory', 'Department'])['Violation_Type'].transform(
                    lambda x: (x != 'Compliant').sum() / len(x) * 100 > 50
                ))
                ]['Employee_Name'].unique()

            st.write(f"Employees in Red Zone: {', '.join(red_zone_employees)}")

        else:
            st.subheader("Critical Zone Insights (Red: < 50% Compliance, Yellow: 50-80%, Green: > 80%)")

            # Calculate compliance rates
            compliance_rates = data.groupby(['Factory', 'Department'])['Violation_Type'].apply(
                lambda x: (x == 'Compliant').sum() / len(x) * 100
            ).reset_index(name='Compliance Rate')

            # Assign zones
            compliance_rates['Zone'] = pd.cut(
                compliance_rates['Compliance Rate'], bins=[0, 50, 80, 100],
                labels=['Red Zone', 'Yellow Zone', 'Green Zone']
            )

            # Display the table
            st.dataframe(compliance_rates)

            # Factory-wise Compliance Chart
            fig_factory_compliance = px.bar(
                compliance_rates,
                x=compliance_rates.groupby('Factory').ngroup() + 1,  # Numeric x-axis
                y='Compliance Rate',
                color='Zone',
                title="Critical Zone Compliance Rates by Factory",
                labels={'x': 'Factory', 'Compliance Rate': 'Compliance Rate (%)'},
                color_discrete_map=zone_colors
            )

            fig_factory_compliance.update_layout(
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                yaxis=dict(range=[0, 100]),
                plot_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(color="white")
            )
            st.plotly_chart(fig_factory_compliance, use_container_width=True)

            # Department-wise Compliance Chart
            fig_department_compliance = px.bar(
                compliance_rates,
                x=compliance_rates.groupby('Department').ngroup() + 1,  # Numeric x-axis
                y='Compliance Rate',
                color='Zone',
                title="Critical Zone Compliance Rates by Department",
                labels={'x': 'Department ', 'Compliance Rate': 'Compliance Rate (%)'},
                color_discrete_map=zone_colors
            )

            fig_department_compliance.update_layout(
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                yaxis=dict(range=[0, 100]),
                plot_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(color="white")
            )
            st.plotly_chart(fig_department_compliance, use_container_width=True)

            # Employees in the Red Zone (Compliance < 50%)
            red_zone_employees = data[
                (data['Violation_Type'] == 'Compliant') &
                (data.groupby(['Factory', 'Department'])['Violation_Type'].transform(
                    lambda x: (x == 'Compliant').sum() / len(x) * 100 < 50
                ))
                ]['Employee_Name'].unique()

            st.write(f"Employees in Red Zone: {', '.join(red_zone_employees)}")

    # Targets Monitoring Insights
    elif selected_insight == "Targets Monitoring Insights":
        # For Violations
        if analysis_type == "Violation":
            st.subheader("Targets Monitoring - Violation Rate vs Targets")

            # Group and calculate violation rates
            violation_data = filtered_data.groupby(['Factory', 'Department']).agg(
                violation_count=('Violation_Type', lambda x: (x != 'Compliant').sum()),
                total=('Violation_Type', 'count')
            ).reset_index()

            violation_data['Target Violation Rate'] = 30  # Assuming target of 30%
            violation_data['Actual Violation Rate'] = (violation_data['violation_count'] / violation_data[
                'total']) * 100

            # Display the data
            st.dataframe(violation_data)

            # ---- Department-wise Chart ----
            st.subheader("Department-wise Actual vs Target Violation Rates")
            fig_dept_violation = px.bar(
                violation_data.melt(id_vars=['Department'],
                                    value_vars=['Actual Violation Rate', 'Target Violation Rate'],
                                    var_name='Rate Type', value_name='Rate (%)'),
                x='Department', y='Rate (%)', color='Rate Type', barmode='group',
                title="Actual vs Target Violation Rates by Department",
                color_discrete_map={'Actual Violation Rate': '#E63946', 'Target Violation Rate': '#F1FAEE'}
            )
            fig_dept_violation.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
            fig_dept_violation.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                                             plot_bgcolor='rgba(0,0,0,0)',
                                             font=dict(color='white'))
            st.plotly_chart(fig_dept_violation, use_container_width=True)

            # ---- Factory-wise Chart ----
            st.subheader("Factory-wise Actual vs Target Violation Rates")
            fig_fact_violation = px.bar(
                violation_data.melt(id_vars=['Factory'],
                                    value_vars=['Actual Violation Rate', 'Target Violation Rate'],
                                    var_name='Rate Type', value_name='Rate (%)'),
                x='Factory', y='Rate (%)', color='Rate Type', barmode='group',
                title="Actual vs Target Violation Rates by Factory",
                color_discrete_map={'Actual Violation Rate': '#F1C40F', 'Target Violation Rate': '#4C5178'}
            )
            fig_fact_violation.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
            fig_fact_violation.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                                             plot_bgcolor='rgba(0,0,0,0)',
                                             font=dict(color='white'))
            st.plotly_chart(fig_fact_violation, use_container_width=True)

        else:
            st.subheader("Targets Monitoring - Compliance Rate vs Targets")

            # Group and calculate compliance rates
            compliance_data = filtered_data.groupby(['Factory', 'Department']).agg(
                compliance_count=('Violation_Type', lambda x: (x == 'Compliant').sum()),
                total=('Violation_Type', 'count')
            ).reset_index()

            compliance_data['Target Compliance Rate'] = 70  # Assuming target of 70%
            compliance_data['Actual Compliance Rate'] = (compliance_data['compliance_count'] / compliance_data[
                'total']) * 100

            # Display the data
            st.dataframe(compliance_data)

            # ---- Department-wise Chart ----
            st.subheader("Department-wise Actual vs Target Compliance Rates")
            fig_dept_compliance = px.bar(
                compliance_data.melt(id_vars=['Department'],
                                     value_vars=['Actual Compliance Rate', 'Target Compliance Rate'],
                                     var_name='Rate Type', value_name='Rate (%)'),
                x='Department', y='Rate (%)', color='Rate Type', barmode='group',
                title="Actual vs Target Compliance Rates by Department",
                color_discrete_map={'Actual Compliance Rate': '#2A9D8F', 'Target Compliance Rate': '#F8EDD8'}
            )
            fig_dept_compliance.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
            fig_dept_compliance.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                                              plot_bgcolor='rgba(0,0,0,0)',
                                              font=dict(color='white'))
            st.plotly_chart(fig_dept_compliance, use_container_width=True)

            # ---- Factory-wise Chart ----
            st.subheader("Factory-wise Actual vs Target Compliance Rates")
            fig_fact_compliance = px.bar(
                compliance_data.melt(id_vars=['Factory'],
                                     value_vars=['Actual Compliance Rate', 'Target Compliance Rate'],
                                     var_name='Rate Type', value_name='Rate (%)'),
                x='Factory', y='Rate (%)', color='Rate Type', barmode='group',
                title="Actual vs Target Compliance Rates by Factory",
                color_discrete_map={'Actual Compliance Rate': '#264653', 'Target Compliance Rate': '#E9C46A'}
            )
            fig_fact_compliance.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
            fig_fact_compliance.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                                              plot_bgcolor='rgba(0,0,0,0)',
                                              font=dict(color='white'))
            st.plotly_chart(fig_fact_compliance, use_container_width=True)

        # Define target rates
        target_violation_rate = 10  # Target for violation (less than 10% violation rate)
        target_compliance_rate = 90  # Target for compliance (more than 90% compliance rate)

        # Custom color mappings for violation and compliance types
        violation_colors = {
            "No Helmet": "#d62728",  # Red
            "No Vest": "#9467bd",  # Purple
            "No Gloves": "#2ca02c",  # Green
            "No Goggles": "#ff7f0e",  # Orange
            "Other": "#1f77b4"  # Blue for any other violation types
        }

        compliance_colors = {
            "Compliant": "#2ca02c",  # Green for compliance
        }

        # Filter data and display insights based on the selected analysis type
        if analysis_type == "Violation":
            st.subheader("Violation Target is 10%")

            # Filter for Violations
            violation_data = data[data['Violation_Type'] != 'Compliant']

            # Create a Difference Chart for Violations
            if not violation_data.empty:
                violation_count = violation_data['Violation_Type'].value_counts().reset_index()
                violation_count.columns = ['Violation Type', 'Count']

                # Current Violation Rate Calculation
                current_violation_rate = (violation_count['Count'].sum() / data.shape[0]) * 100 if data.shape[
                                                                                                       0] > 0 else 0

                # Create a Difference Chart
                fig_difference = go.Figure()
                fig_difference.add_trace(go.Bar(
                    x=['Current Rate', 'Target Rate'],
                    y=[current_violation_rate, target_violation_rate],
                    name='Rates',
                    marker_color=['#1f77b4', '#ff7f0e'],  # Blue for current rate, orange for target
                ))

                # Add Line for Difference
                fig_difference.add_trace(go.Scatter(
                    x=['Current Rate', 'Target Rate'],
                    y=[current_violation_rate, target_violation_rate],
                    mode='lines+text',
                    name='Difference',
                    text=[f"{current_violation_rate:.2f}%", f"{target_violation_rate:.2f}%"],
                    textposition='top center',
                    line=dict(color='red', width=2)  # Red line for difference
                ))

                # Update layout for the difference chart
                fig_difference.update_layout(
                    title='Current vs. Target Violation Rate',
                    xaxis_title='Rate Type',
                    yaxis_title='Rate (%)',
                    showlegend=True,
                    plot_bgcolor='rgba(0, 0, 0, 0)'  # Transparent background
                )

                # Display Difference Chart
                st.plotly_chart(fig_difference, use_container_width=True)

                # Create a Bar Chart for Violation Counts with custom colors
                violation_count['Color'] = violation_count['Violation Type'].map(violation_colors)
                fig_violation = px.bar(violation_count, x='Violation Type', y='Count',
                                       title="Violation Counts",
                                       labels={'Count': 'Number of Violations'},
                                       color='Violation Type',
                                       color_discrete_map=violation_colors)  # Assign specific colors to each violation type

                # Update layout for violation chart
                fig_violation.update_layout(
                    plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                    font=dict(color="white"),  # White font for contrast
                )

                # Display Violation Chart
                st.plotly_chart(fig_violation, use_container_width=True)

            else:
                st.write("No violation data available for the selected filters.")

        elif analysis_type == "Compliance":
            st.subheader("Compliance Target is 90%")

            # Filter for Compliance
            compliance_data = data[data['Violation_Type'] == 'Compliant']

            # Create a Difference Chart for Compliance
            if not compliance_data.empty:
                compliance_count = compliance_data['Violation_Type'].value_counts().reset_index()
                compliance_count.columns = ['Compliance Type', 'Count']

                # Current Compliance Rate Calculation
                current_compliance_rate = (compliance_count['Count'].sum() / data.shape[0]) * 100 if data.shape[
                                                                                                         0] > 0 else 0

                # Create a Difference Chart for Compliance
                fig_difference_compliance = go.Figure()
                fig_difference_compliance.add_trace(go.Bar(
                    x=['Current Rate', 'Target Rate'],
                    y=[current_compliance_rate, target_compliance_rate],
                    name='Rates',
                    marker_color=['#2ca02c', '#ff7f0e'],  # Green for current rate, orange for target
                ))

                # Add Line for Difference
                fig_difference_compliance.add_trace(go.Scatter(
                    x=['Current Rate', 'Target Rate'],
                    y=[current_compliance_rate, target_compliance_rate],
                    mode='lines+text',
                    name='Difference',
                    text=[f"{current_compliance_rate:.2f}%", f"{target_compliance_rate:.2f}%"],
                    textposition='top center',
                    line=dict(color='red', width=2)  # Red line for difference
                ))

                # Update layout for compliance difference chart
                fig_difference_compliance.update_layout(
                    title='Current vs. Target Compliance Rate',
                    xaxis_title='Rate Type',
                    yaxis_title='Rate (%)',
                    showlegend=True,
                    plot_bgcolor='rgba(0, 0, 0, 0)'  # Transparent background
                )

                # Display Difference Chart
                st.plotly_chart(fig_difference_compliance, use_container_width=True)

                # Create a Bar Chart for Compliance Counts with custom colors
                compliance_count['Color'] = compliance_count['Compliance Type'].map(compliance_colors)
                fig_compliance = px.bar(compliance_count, x='Compliance Type', y='Count',
                                        title="Compliance Counts",
                                        labels={'Count': 'Number of Compliant Cases'},
                                        color='Compliance Type',
                                        color_discrete_map=compliance_colors)  # Assign specific colors to compliance

                # Update layout for compliance chart
                fig_compliance.update_layout(
                    plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                    font=dict(color="white"),  # White font for contrast
                )

                # Display Compliance Chart
                st.plotly_chart(fig_compliance, use_container_width=True)

            else:
                st.write("No compliance data available for the selected filters.")


    # Time Tracking Insights
    elif selected_insight == "Time Tracking Insights":

        filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])

        # Group data by Date and calculate violation/compliance rates over time
        if analysis_type == "Violation":
            st.subheader("Time Tracking Insights - Violation Rate Over Time")

            # Calculate daily violation rates
            time_tracking_violations = filtered_data.groupby('Date')['Violation_Type'].apply(
                lambda x: (x != 'Compliant').sum() / len(x) * 100).reset_index(name='Violation Rate')

            st.dataframe(time_tracking_violations)

            # Line chart to track violations over time
            fig_time_tracking_violations = px.line(time_tracking_violations,
                                                   x='Date',
                                                   y='Violation Rate',
                                                   title="Violation Rate Over Time",
                                                   labels={'Violation Rate': 'Violation Rate (%)'},
                                                   line_shape='spline',  # Smooth curve
                                                   markers=True,  # Show markers on data points
                                                   color_discrete_sequence=px.colors.qualitative.Dark24)

            fig_time_tracking_violations.update_layout(
                xaxis_title="Date",
                yaxis_title="Violation Rate (%)",
                yaxis=dict(range=[0, 100])  # Set Y-axis range from 0 to 100%
            )

            st.plotly_chart(fig_time_tracking_violations, use_container_width=True)

        else:
            st.subheader("Time Tracking Insights - Compliance Rate Over Time")

            # Calculate daily compliance rates
            time_tracking_compliance = filtered_data.groupby('Date')['Violation_Type'].apply(
                lambda x: (x == 'Compliant').sum() / len(x) * 100).reset_index(name='Compliance Rate')

            st.dataframe(time_tracking_compliance)

            # Line chart to track compliance over time
            fig_time_tracking_compliance = px.line(time_tracking_compliance,
                                                   x='Date',
                                                   y='Compliance Rate',
                                                   title="Compliance Rate Over Time",
                                                   labels={'Compliance Rate': 'Compliance Rate (%)'},
                                                   line_shape='spline',  # Smooth curve
                                                   markers=True,  # Show markers on data points
                                                   color_discrete_sequence=px.colors.qualitative.Dark24)

            fig_time_tracking_compliance.update_layout(
                xaxis_title="Date",
                yaxis_title="Compliance Rate (%)",
                yaxis=dict(range=[0, 100])  # Set Y-axis range from 0 to 100%
            )

            st.plotly_chart(fig_time_tracking_compliance, use_container_width=True)


    # Shift Productivity Insights
    elif selected_insight == "Shift Productivity Insights":
        st.subheader("Shift Productivity Insights")

        # Shift Productivity Insights for Violations and Compliance


        # Group data by Shift, Factory, and Department for both Violations and Compliance
        if analysis_type == "Violation":
            st.subheader("Shift Productivity Insights - Violation Rate by Shift")

            # Calculate violation rates for each shift
            shift_violations = filtered_data.groupby(['Shift', 'Factory', 'Department'])['Violation_Type'].apply(
                lambda x: (x != 'Compliant').sum() / len(x) * 100).reset_index(name='Violation Rate')

            st.dataframe(shift_violations)

            # ---- Violation Rate by Shift & Department (Line Chart) ----
            st.subheader("Violation Rate by Shift & Department")
            fig_violation_dept = px.line(
                shift_violations,
                x='Shift',
                y='Violation Rate',
                color='Department',
                markers=True,  # Add markers for clarity
                title="Violation Rate by Shift & Department",
                labels={'Violation Rate': 'Violation Rate (%)'},
                line_shape='spline',  # Smooth lines for better aesthetics
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_violation_dept.update_layout(
                xaxis_title="Shift",
                yaxis_title="Violation Rate (%)",
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_violation_dept, use_container_width=True)

            # ---- Violation Rate by Shift & Factory (Line Chart) ----
            st.subheader("Violation Rate by Shift & Factory")
            fig_violation_factory = px.line(
                shift_violations,
                x='Shift',
                y='Violation Rate',
                color='Factory',
                markers=True,
                title="Violation Rate by Shift & Factory",
                labels={'Violation Rate': 'Violation Rate (%)'},
                line_shape='spline',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_violation_factory.update_layout(
                xaxis_title="Shift",
                yaxis_title="Violation Rate (%)",
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_violation_factory, use_container_width=True)

        else:
            st.subheader("Shift Productivity Insights - Compliance Rate by Shift")

            # Calculate compliance rates for each shift
            shift_compliance = filtered_data.groupby(['Shift', 'Factory', 'Department'])['Violation_Type'].apply(
                lambda x: (x == 'Compliant').sum() / len(x) * 100).reset_index(name='Compliance Rate')

            st.dataframe(shift_compliance)

            # ---- Compliance Rate by Shift & Department (Line Chart) ----
            st.subheader("Compliance Rate by Shift & Department")
            fig_compliance_dept = px.line(
                shift_compliance,
                x='Shift',
                y='Compliance Rate',
                color='Department',
                markers=True,
                title="Compliance Rate by Shift & Department",
                labels={'Compliance Rate': 'Compliance Rate (%)'},
                line_shape='spline',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_compliance_dept.update_layout(
                xaxis_title="Shift",
                yaxis_title="Compliance Rate (%)",
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_compliance_dept, use_container_width=True)

            # ---- Compliance Rate by Shift & Factory (Line Chart) ----
            st.subheader("Compliance Rate by Shift & Factory")
            fig_compliance_factory = px.line(
                shift_compliance,
                x='Shift',
                y='Compliance Rate',
                color='Factory',
                markers=True,
                title="Compliance Rate by Shift & Factory",
                labels={'Compliance Rate': 'Compliance Rate (%)'},
                line_shape='spline',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_compliance_factory.update_layout(
                xaxis_title="Shift",
                yaxis_title="Compliance Rate (%)",
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_compliance_factory, use_container_width=True)


        # Group data by Shift, Factory, and Department to analyze productivity for each shift
        if analysis_type == "Violation":
            st.subheader("Shift Productivity Insights - Violation Rate by Shift")

            # Calculate violation rates for each shift
            shift_violations = filtered_data.groupby(['Shift', 'Factory', 'Department'])['Violation_Type'].apply(
                lambda x: (x != 'Compliant').sum() / len(x) * 100).reset_index(name='Violation Rate')



            # Bar chart to visualize violation rates by shift
            fig_shift_violations = px.bar(shift_violations,
                                          x='Shift',
                                          y='Violation Rate',
                                          color='Factory',
                                          facet_col='Department',  # Separate charts for each department
                                          title="Violation Rate by Shift",
                                          labels={'Violation Rate': 'Violation Rate (%)'},
                                          color_discrete_sequence=px.colors.qualitative.Pastel)

            fig_shift_violations.update_layout(
                xaxis_title="Shift",
                yaxis_title="Violation Rate (%)",
                yaxis=dict(range=[0, 100]),  # Set Y-axis range from 0 to 100%
            )

            st.plotly_chart(fig_shift_violations, use_container_width=True)

        else:
            st.subheader("Shift Productivity Insights - Compliance Rate by Shift")

            # Calculate compliance rates for each shift
            shift_compliance = filtered_data.groupby(['Shift', 'Factory', 'Department'])['Violation_Type'].apply(
                lambda x: (x == 'Compliant').sum() / len(x) * 100).reset_index(name='Compliance Rate')



            # Bar chart to visualize compliance rates by shift
            fig_shift_compliance = px.bar(shift_compliance,
                                          x='Shift',
                                          y='Compliance Rate',
                                          color='Factory',
                                          facet_col='Department',  # Separate charts for each department
                                          title="Compliance Rate by Shift",
                                          labels={'Compliance Rate': 'Compliance Rate (%)'},
                                          color_discrete_sequence=px.colors.qualitative.Pastel)

            fig_shift_compliance.update_layout(
                xaxis_title="Shift",
                yaxis_title="Compliance Rate (%)",
                yaxis=dict(range=[0, 100]),  # Set Y-axis range from 0 to 100%
            )

            st.plotly_chart(fig_shift_compliance, use_container_width=True)



    # Growth Tracker Insights
    elif selected_insight == "Growth Tracker Insights":

        # Convert 'Date' column to datetime (adjust column name if needed)
        data['Date'] = pd.to_datetime(data['Date'])
        # Sidebar: Interval Selection (Day, Month, or Year)
        interval = st.radio("Select Interval:", ["Day", "Month", "Year"])
        # Group data based on the selected interval
        if interval == "Day":
            data['Interval'] = data['Date'].dt.date
        elif interval == "Month":
            data['Interval'] = data['Date'].dt.to_period('M').astype(str)
        else:
            data['Interval'] = data['Date'].dt.year

        # Filter the data based on the analysis type
        if analysis_type == "Violation":
            filtered_data = data[data['Violation_Type'] != 'Compliant']
        else:
            filtered_data = data[data['Violation_Type'] == 'Compliant']


        total_counts = filtered_data.shape[0]  # Total number of entries in filtered_data


        grouped_data = (
            filtered_data.groupby(['Interval', 'Factory', 'Department', 'Shift']).size().reset_index(name='Count')
        )

        # Check if 'Count' is numeric and convert to numeric type
        grouped_data['Count'] = pd.to_numeric(grouped_data['Count'], errors='coerce')

        # Calculate Current Rate
        grouped_data['Current Rate'] = (grouped_data['Count'] / total_counts) * 100  # Ensure we are using numeric

        # Drop any rows where Current Rate is NaN (if any)
        grouped_data = grouped_data.dropna(subset=['Current Rate'])

        # Calculate Growth Rate for each interval
        grouped_data['Growth Rate'] = grouped_data.groupby(['Factory', 'Department', 'Shift'])[
            'Current Rate'].diff().fillna(0)


        # Display the growth tracker table
        st.subheader(f"Growth Tracker Insights - {analysis_type} By Units ")
        st.dataframe(grouped_data)

        # Filter data for compliant or violation checks
        if analysis_type == "Violation":
            filtered_data = data[data['Violation_Type'] != 'Compliant']
        else:
            filtered_data = data[data['Violation_Type'] == 'Compliant']

        interval_counts = filtered_data.groupby('Interval').size().reset_index(name='Filtered Count')

        # Group by 'Interval' for the total counts (if necessary, else use len(data))
        total_counts = data.groupby('Interval').size().reset_index(name='Total Count')

        # Merge the counts to calculate the Current Rate
        interval_rate = pd.merge(interval_counts, total_counts, on='Interval', how='outer')

        # Calculate Current Rate as a percentage
        interval_rate['Current Rate'] = (interval_rate['Filtered Count'] / interval_rate['Total Count']) * 100

        # Fill NaN values with 0 for Current Rate
        interval_rate['Current Rate'] = interval_rate['Current Rate'].fillna(0)

        # Calculate growth between intervals
        interval_rate['Growth Rate'] = interval_rate['Current Rate'].diff().fillna(0)



        # Calculate total growth
        total_growth = interval_rate['Growth Rate'].sum()
        trend_message = f"Overall {analysis_type} Growth: {total_growth:.2f}% over selected period."

        # Display growth tracker insights
        st.subheader(f"Growth Tracker Insights - {analysis_type}  Over All")
        st.dataframe(interval_rate)

        # Plot the growth rate chart
        fig_growth = px.line(
            interval_rate,
            x='Interval',
            y='Current Rate',
            title=f"{analysis_type} Rate Over Time ({interval}-wise)",
            labels={'Current Rate': f'{analysis_type} Rate (%)'},
            markers=True,
            color_discrete_sequence=['green' if total_growth > 0 else 'red']
        )

        # Update layout for better visualization
        fig_growth.update_layout(
            xaxis_title=f"{interval}",
            yaxis_title=f"{analysis_type} Rate (%)",
            yaxis=dict(range=[0, 100])
        )

        # Display the chart
        st.plotly_chart(fig_growth, use_container_width=True)


    # Risk Radar Insights
    elif selected_insight == "Risk Radar Insights":
        # Convert 'Date' column to datetime (adjust column name if needed)
        data['Date'] = pd.to_datetime(data['Date'])
        # Sidebar: Interval Selection (Day, Month, or Year)
        interval = st.radio("Select Interval:", ["Day", "Month", "Year"])


        # Group data based on the selected interval
        if interval == "Day":
            data['Interval'] = data['Date'].dt.date
            violation_threshold = 20  # Day threshold for violation
            compliance_threshold = 80  # Day threshold for compliance
        elif interval == "Month":
            data['Interval'] = data['Date'].dt.to_period('M').astype(str)
            violation_threshold = 30  # Month threshold for violation
            compliance_threshold = 70  # Month threshold for compliance
        else:
            data['Interval'] = data['Date'].dt.year
            violation_threshold = 60  # Year threshold for violation
            compliance_threshold = 40  # Year threshold for compliance

        # Filter the data based on the analysis type
        if analysis_type == "Violation":
            filtered_data = data[data['Violation_Type'] != 'Compliant']
        else:
            filtered_data = data[data['Violation_Type'] == 'Compliant']

        # Calculate total counts and filtered counts per interval
        interval_counts = filtered_data.groupby('Interval').size().reset_index(name='Filtered Count')
        total_counts = data.groupby('Interval').size().reset_index(name='Total Count')

        # Merge counts to calculate the Current Rate
        interval_rate = pd.merge(interval_counts, total_counts, on='Interval', how='outer')
        interval_rate['Current Rate'] = (interval_rate['Filtered Count'] / interval_rate['Total Count']) * 100
        interval_rate['Current Rate'] = interval_rate['Current Rate'].fillna(0)  # Fill NaN with 0

        # Calculate Growth Rate between intervals
        interval_rate['Growth Rate'] = interval_rate['Current Rate'].diff().fillna(0)


        # Determine dot color based on the threshold logic
        def get_dot_color(growth):
            if analysis_type == "Violation":
                if (interval == "Day" and growth > violation_threshold) or \
                        (interval == "Month" and growth > violation_threshold) or \
                        (interval == "Year" and growth > violation_threshold):
                    return 'red'
            else:  # Compliance
                if (interval == "Day" and growth < compliance_threshold) or \
                        (interval == "Month" and growth < compliance_threshold) or \
                        (interval == "Year" and growth < compliance_threshold):
                    return 'red'
            return 'green'


        # Apply the color logic to each interval's growth rate
        interval_rate['Dot Color'] = interval_rate['Growth Rate'].apply(get_dot_color)

        # Display growth tracker insights
        st.subheader(f"Growth Tracker Insights - {analysis_type} Over All")
        st.dataframe(interval_rate)

        # Plot the growth rate line chart with colored dots
        fig_growth = px.line(
            interval_rate,
            x='Interval',
            y='Current Rate',
            title=f"{analysis_type} Rate Over Time ({interval}-wise)",
            labels={'Current Rate': f'{analysis_type} Rate (%)'},
            markers=True
        )

        # Add colored markers (dots) based on growth rate thresholds
        fig_growth.update_traces(
            marker=dict(size=10),
            marker_color=interval_rate['Dot Color']
        )

        # Update layout for better visualization
        fig_growth.update_layout(
            xaxis_title=f"{interval}",
            yaxis_title=f"{analysis_type} Rate (%)",
            yaxis=dict(range=[0, 100])  # Y-axis range from 0 to 100%
        )

        # Display the chart
        st.plotly_chart(fig_growth, use_container_width=True)
