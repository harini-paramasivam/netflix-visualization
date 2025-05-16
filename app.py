import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# Define the Netflix color theme palette for all visualizations
RED_PALETTE = ["#E50914", "#B81D24", "#831010", "#E87C03", "#F5F5F1", "#B31217", "#D81F26", "#F1041C"]
RED_SEQUENTIAL = ["#FFEBEE", "#FFCDD2", "#EF9A9A", "#E57373", "#EF5350", "#F44336", "#E53935", "#D32F2F", "#C62828", "#B71C1C"]
RED_BASE = "#E50914"  # Netflix red
DARK_RED = "#831010"  # Darker Netflix red
BACKGROUND_COLOR = "#121212"  # Dark background
CARD_BACKGROUND = "#1F1F1F"  # Slightly lighter background for cards

# Create a function to generate sample data for demonstration
@st.cache_data
def load_sample_data():
    # Create sample data with genres, countries, release years, etc.
    n_samples = 3000
    
    # Genres and their weights for realistic distribution
    genres = ['Action & Adventure', 'Comedy', 'Drama', 'Horror', 'Thriller', 'Romance', 
              'Documentary', 'Sci-Fi', 'Family', 'Fantasy', 'Crime', 'Animation']
    genre_weights = [0.18, 0.16, 0.15, 0.1, 0.08, 0.08, 0.07, 0.06, 0.05, 0.03, 0.02, 0.02]
    
    # Countries with realistic distribution
    countries = ['United States', 'India', 'United Kingdom', 'Japan', 'South Korea', 
                'France', 'Canada', 'Spain', 'Germany', 'Australia', 'Brazil', 'Mexico']
    country_weights = [0.35, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05, 0.04, 0.04, 0.03, 0.02, 0.02]
    
    # Content types and weights
    types = ['Movie', 'TV Show']
    type_weights = [0.7, 0.3]
    
    # Ratings and weights
    ratings = ['TV-MA', 'R', 'PG-13', 'TV-14', 'TV-PG', 'PG', 'G', 'TV-Y7', 'TV-Y']
    rating_weights = [0.25, 0.20, 0.15, 0.15, 0.10, 0.05, 0.04, 0.03, 0.03]
    
    # Generate realistic release years (more content in recent years)
    years = list(range(2010, 2023))
    year_weights = [(y - 2009) ** 2 for y in years]
    year_weights = [w / sum(year_weights) for w in year_weights]
    
    # Generate data
    release_years = np.random.choice(years, n_samples, p=year_weights)
    types_data = np.random.choice(types, n_samples, p=type_weights)
    
    # Structure for sample data
    data = {
        'title': [f'Title {i}' for i in range(1, n_samples + 1)],
        'type': types_data,
        'release_year': release_years,
        'genre': np.random.choice(genres, n_samples, p=genre_weights),
        'country': np.random.choice(countries, n_samples, p=country_weights),
        'rating': np.random.choice(ratings, n_samples, p=rating_weights),
        'duration': [f'{np.random.randint(70, 180)} min' if t == 'Movie' else f'{np.random.randint(1, 8)} Seasons' 
                    for t in types_data],
        'views': np.random.randint(10000, 5000000, n_samples),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Extract duration as numeric for analysis
    df['duration_value'] = df['duration'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
    
    # Add a year column for easier filtering
    df['year'] = df['release_year']
    
    return df

# Page configuration
st.set_page_config(
    page_title="Netflix Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for styling to match the example image
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E50914;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .subheader {
        font-size: 1.2rem;
        color: #999;
        margin-bottom: 2rem;
        text-align: center;
    }
    .stApp {
        background-color: #000000;
    }
    .chart-container {
        background-color: #141414;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #141414;
        border-radius: 5px;
        padding: 1rem;
        text-align: center;
        height: 100%;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #E50914;
    }
    .metric-label {
        font-size: 1rem;
        color: #DDD;
    }
    div[data-testid="stSidebar"] {
        background-color: #141414;
        border-right: 1px solid #333;
    }
    div.stSlider > div {
        background-color: #333;
    }
    div.stSlider > div > div > div {
        background-color: #E50914;
    }
    .netflix-logo {
        font-size: 3rem;
        font-weight: bold;
        color: #E50914;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: 4px;
    }
    .filter-section {
        background-color: #141414;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .filter-title {
        color: #E50914;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    button {
        background-color: #E50914;
        color: white;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFF;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #141414;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E50914;
    }
</style>
""", unsafe_allow_html=True)

# Load data
df = load_sample_data()

# Sidebar filters
with st.sidebar:
    st.markdown('<div class="netflix-logo">NETFLIX</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="filter-title">Content Filters</div>', unsafe_allow_html=True)
    
    # Filter by content type
    content_type = st.selectbox(
        "Content Type",
        ["All"] + sorted(df['type'].unique().tolist())
    )
    
    # Filter by genre
    available_genres = sorted(df['genre'].unique().tolist())
    selected_genres = st.multiselect(
        "Genres",
        available_genres,
        default=[]
    )
    
    # Filter by year range
    min_year = int(df['release_year'].min())
    max_year = int(df['release_year'].max())
    year_range = st.slider(
        "Release Years",
        min_year, max_year, (min_year, max_year)
    )
    
    # Filter by countries
    available_countries = sorted(df['country'].unique().tolist())
    selected_countries = st.multiselect(
        "Countries",
        available_countries,
        default=[]
    )
    
    # Apply button
    apply_filters = st.button("Apply Filters")

# Filter data based on sidebar inputs
filtered_df = df.copy()

if content_type != "All":
    filtered_df = filtered_df[filtered_df['type'] == content_type]

if selected_genres:
    filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]

filtered_df = filtered_df[
    (filtered_df['release_year'] >= year_range[0]) & 
    (filtered_df['release_year'] <= year_range[1])
]

if selected_countries:
    filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]

# Main dashboard layout
st.markdown('<div class="netflix-logo">NETFLIX</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Content Dashboard | 2023</div>', unsafe_allow_html=True)

# Top stats row
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_titles = len(filtered_df)
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{total_titles:,}</div>
        <div class="metric-label">Total Titles</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    movies_count = filtered_df[filtered_df['type'] == 'Movie'].shape[0]
    tv_count = filtered_df[filtered_df['type'] == 'TV Show'].shape[0]
    ratio = f"{round(movies_count / total_titles * 100)}% / {round(tv_count / total_titles * 100)}%"
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{ratio}</div>
        <div class="metric-label">Movies / TV Shows</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    total_countries = filtered_df['country'].nunique()
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{total_countries}</div>
        <div class="metric-label">Countries</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    top_genre = filtered_df['genre'].value_counts().index[0]
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{top_genre}</div>
        <div class="metric-label">Top Genre</div>
    </div>
    """, unsafe_allow_html=True)

# Main content - left side: map, right side: charts
row1_col1, row1_col2 = st.columns([1, 1])

with row1_col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Content Distribution by Country")
    
    # Create choropleth map
    country_counts = filtered_df['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    
    fig = px.choropleth(
        country_counts,
        locations='country',
        locationmode='country names',
        color='count',
        color_continuous_scale=RED_SEQUENTIAL,
        labels={'count': 'Number of Titles'},
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth',
            bgcolor=BACKGROUND_COLOR,
            landcolor='#1c1c1c',
            coastlinecolor='#555',
            countrycolor='#333',
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color='white'),
        height=300,
        coloraxis_colorbar=dict(
            title="Titles",
            thicknessmode="pixels", 
            thickness=15,
            len=0.7,
            title_font=dict(color='white'),
            tickfont=dict(color='white')
        ),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add genre bar chart below map
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Top Genres")
    
    genre_counts = filtered_df['genre'].value_counts().nlargest(10).reset_index()
    genre_counts.columns = ['genre', 'count']
    
    fig = px.bar(
        genre_counts,
        x='count',
        y='genre',
        orientation='h',
        color_discrete_sequence=[RED_BASE],
        labels={'count': 'Number of Titles', 'genre': 'Genre'},
    )
    
    fig.update_layout(
        margin=dict(l=0, r=10, t=10, b=0),
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color='white'),
        height=300,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            title=dict(text="Number of Titles", font=dict(color='white')),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            title=dict(text="", font=dict(color='white')),
            autorange="reversed",
        ),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with row1_col2:
    # Create content type pie chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Content Type Distribution")
    
    type_counts = filtered_df['type'].value_counts().reset_index()
    type_counts.columns = ['type', 'count']
    
    fig = go.Figure(data=[go.Pie(
        labels=type_counts['type'],
        values=type_counts['count'],
        hole=.7,
        marker=dict(colors=[RED_BASE, '#B81D24']),
        textinfo='percent',
        textfont=dict(color='white', size=14),
    )])
    
    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=10, b=0),
        height=300,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(color='white')
        ),
        annotations=[
            dict(
                text=f"{total_titles:,}<br>Titles",
                x=0.5, y=0.5,
                font=dict(size=20, color='white'),
                showarrow=False
            )
        ]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create release year trend line
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Titles by Release Year")
    
    yearly_counts = filtered_df.groupby(['release_year', 'type']).size().reset_index(name='count')
    
    fig = px.line(
        yearly_counts,
        x='release_year',
        y='count',
        color='type',
        color_discrete_map={'Movie': RED_BASE, 'TV Show': '#B81D24'},
        labels={'count': 'Number of Titles', 'release_year': 'Year', 'type': 'Type'},
    )
    
    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=10, b=0),
        height=300,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            title=dict(text="Year", font=dict(color='white')),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            title=dict(text="Number of Titles", font=dict(color='white')),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(color='white')
        ),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Bottom row with ratings distribution
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.subheader("Content Rating Distribution")

rating_counts = filtered_df.groupby(['rating', 'type']).size().reset_index(name='count')

fig = px.bar(
    rating_counts,
    x='rating',
    y='count',
    color='type',
    barmode='group',
    color_discrete_map={'Movie': RED_BASE, 'TV Show': '#B81D24'},
    labels={'count': 'Number of Titles', 'rating': 'Content Rating', 'type': 'Type'},
)

fig.update_layout(
    paper_bgcolor=BACKGROUND_COLOR,
    plot_bgcolor=BACKGROUND_COLOR,
    font=dict(color='white'),
    margin=dict(l=0, r=0, t=10, b=0),
    height=300,
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        title=dict(text="Content Rating", font=dict(color='white')),
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        title=dict(text="Number of Titles", font=dict(color='white')),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5,
        font=dict(color='white')
    ),
)

st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #999; padding: 10px;">
    Netflix Data Dashboard | Created with Streamlit | Data as of 2023
</div>
""", unsafe_allow_html=True)