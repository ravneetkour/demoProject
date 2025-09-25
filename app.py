import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random
import time

# Set page config
st.set_page_config(
    page_title="DermaCare AI - Smart Skincare Solutions",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern web styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Poppins', sans-serif;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.9;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        padding: 2rem;
        border-radius: 20px;
        color: #333;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .step-card {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .brand-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }
    
    .brand-card:hover {
        transform: scale(1.02);
    }
    
    .profile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        border-top: 3px solid #667eea;
    }
    
    .section-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #2d3748;
        text-align: center;
        margin: 3rem 0 2rem 0;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 4px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
    }
    
    .tip-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .testimonial-card {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }
    
    .navbar {
        background: white;
        padding: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .progress-bar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        height: 8px;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    .sidebar .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
    }
    
    .sidebar .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .sidebar .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Generate comprehensive skincare dataset with more realistic data
@st.cache_data
def generate_skincare_dataset():
    np.random.seed(42)
    random.seed(42)
    
    # Define categories with more options
    skin_types = ['Oily', 'Dry', 'Combination', 'Sensitive', 'Normal']
    age_groups = ['Teens (13-19)', 'Young Adult (20-29)', 'Adult (30-39)', 'Mature (40+)']
    concerns = ['Acne & Breakouts', 'Fine Lines & Wrinkles', 'Dark Spots & Hyperpigmentation', 
               'Dryness & Dehydration', 'Sensitivity & Redness', 'Excess Oil & Large Pores', 
               'Dullness & Uneven Texture', 'Loss of Firmness']
    climates = ['Humid & Hot', 'Dry & Arid', 'Moderate & Temperate', 'Cold & Windy']
    lifestyles = ['Very Active (Exercise 5+ times/week)', 'Moderately Active (Exercise 2-4 times/week)', 
                 'Lightly Active (Exercise 1-2 times/week)', 'Sedentary (Minimal exercise)']
    
    # Enhanced skincare routines with detailed steps
    routines = {
        'Essential Basics': {
            'steps': ['Gentle Foaming Cleanser', 'Hydrating Toner', 'Daily Moisturizer', 'Broad Spectrum SPF 30+'],
            'description': 'Perfect for beginners or those wanting a simple, effective routine'
        },
        'Acne Combat System': {
            'steps': ['Salicylic Acid Cleanser', 'BHA Exfoliating Toner', 'Niacinamide Serum', 'Oil-Free Moisturizer', 'Non-Comedogenic Sunscreen'],
            'description': 'Targets breakouts, blackheads, and controls excess oil production'
        },
        'Anti-Aging Protocol': {
            'steps': ['Gentle Cream Cleanser', 'Vitamin C Serum (AM)', 'Retinol Treatment (PM)', 'Peptide Moisturizer', 'Anti-Aging Sunscreen SPF 50'],
            'description': 'Reduces fine lines, boosts collagen, and prevents future aging'
        },
        'Hydration Intensive': {
            'steps': ['Cream Cleanser', 'Hyaluronic Acid Toner', 'Vitamin E Serum', 'Rich Night Cream', 'Hydrating Face Oil', 'Moisturizing Sunscreen'],
            'description': 'Deep hydration for dry, dehydrated, or mature skin'
        },
        'Sensitive Skin Care': {
            'steps': ['Micellar Water Cleanser', 'Alcohol-Free Toner', 'Ceramide Serum', 'Fragrance-Free Moisturizer', 'Mineral Sunscreen'],
            'description': 'Gentle, calming ingredients for reactive and sensitive skin'
        },
        'Brightening Complex': {
            'steps': ['Gentle Exfoliating Cleanser', 'Vitamin C Serum', 'Niacinamide Treatment', 'Brightening Moisturizer', 'Antioxidant Sunscreen'],
            'description': 'Fades dark spots, evens skin tone, and adds radiant glow'
        },
        'Oil Control Matrix': {
            'steps': ['Deep Cleansing Foam', 'Salicylic Acid Toner', 'Zinc Serum', 'Lightweight Gel Moisturizer', 'Mattifying Sunscreen'],
            'description': 'Controls shine, minimizes pores, and prevents breakouts'
        }
    }
    
    # Enhanced brand recommendations with price points
    brands = {
        'Budget-Friendly ($5-$20)': {
            'brands': ['CeraVe', 'The Ordinary', 'Neutrogena', 'Aveeno', 'Olay', 'La Roche-Posay'],
            'description': 'Quality skincare without breaking the bank'
        },
        'Mid-Range ($20-$60)': {
            'brands': ['Paula\'s Choice', 'Drunk Elephant', 'Glossier', 'Tatcha', 'Fresh', 'Kiehl\'s'],
            'description': 'Premium ingredients with proven results'
        },
        'Luxury ($60+)': {
            'brands': ['SK-II', 'La Mer', 'Estée Lauder', 'Clinique', 'Dior', 'Chanel'],
            'description': 'High-end formulations with luxury experience'
        }
    }
    
    # Generate more realistic dataset
    data = []
    for i in range(10000):  # Increased dataset size
        skin_type = random.choice(skin_types)
        age_group = random.choice(age_groups)
        primary_concern = random.choice(concerns)
        climate = random.choice(climates)
        lifestyle = random.choice(lifestyles)
        budget = random.choice(list(brands.keys()))
        
        # Enhanced logic for routine recommendation
        if 'Acne' in primary_concern and skin_type == 'Oily':
            routine = 'Acne Combat System'
        elif 'Dry' in skin_type or 'Dehydration' in primary_concern:
            routine = 'Hydration Intensive'
        elif 'Mature' in age_group or 'Wrinkles' in primary_concern:
            routine = 'Anti-Aging Protocol'
        elif skin_type == 'Sensitive' or 'Sensitivity' in primary_concern:
            routine = 'Sensitive Skin Care'
        elif 'Dark Spots' in primary_concern or 'Dullness' in primary_concern:
            routine = 'Brightening Complex'
        elif skin_type == 'Oily' and 'Pores' in primary_concern:
            routine = 'Oil Control Matrix'
        else:
            routine = 'Essential Basics'
        
        # Get routine details
        routine_steps = routines[routine]['steps']
        routine_description = routines[routine]['description']
        brand_info = brands[budget]
        recommended_brands = random.sample(brand_info['brands'], min(3, len(brand_info['brands'])))
        
        data.append({
            'skin_type': skin_type,
            'age_group': age_group,
            'primary_concern': primary_concern,
            'climate': climate,
            'lifestyle': lifestyle,
            'budget': budget,
            'recommended_routine': routine,
            'routine_description': routine_description,
            'routine_steps': ' → '.join(routine_steps),
            'recommended_brands': ', '.join(recommended_brands),
            'brand_description': brand_info['description']
        })
    
    return pd.DataFrame(data)

# Train enhanced ML model
@st.cache_data
def train_model(df):
    # Prepare features for ML
    encoders = {}
    feature_columns = ['skin_type', 'age_group', 'primary_concern', 'climate', 'lifestyle', 'budget']
    
    X_encoded = pd.DataFrame()
    for col in feature_columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    le_routine = LabelEncoder()
    y = le_routine.fit_transform(df['recommended_routine'])
    encoders['routine'] = le_routine
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train enhanced model
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, encoders, accuracy

# Skincare tips and education
def get_skincare_tips():
    tips = [
        {
            'title': '☀️ Never Skip Sunscreen',
            'content': 'Apply SPF 30+ daily, even indoors. UV rays cause 80% of visible aging signs.'
        },
        {
            'title': '💧 Hydration is Key',
            'content': 'Drink 8 glasses of water daily and use a humidifier to maintain skin moisture.'
        },
        {
            'title': '😴 Beauty Sleep is Real',
            'content': 'Your skin repairs itself during sleep. Aim for 7-9 hours nightly.'
        },
        {
            'title': '🧴 Less is More',
            'content': 'Don\'t overwhelm your skin. Introduce new products gradually, one at a time.'
        },
        {
            'title': '🥗 Nutrition Matters',
            'content': 'Omega-3 fatty acids, antioxidants, and vitamins A, C, E support healthy skin.'
        }
    ]
    return tips

# Main application
def main():
    # Hero Section
    st.markdown('''
    <div class="hero-section">
        <div class="hero-title">✨ DermaCare AI</div>
        <div class="hero-subtitle">Your Personal Skincare Expert Powered by Advanced AI</div>
        <p style="font-size: 1.1rem; margin-top: 1rem;">
            Get personalized skincare recommendations based on cutting-edge machine learning 
            and dermatological expertise. Transform your skin with science-backed solutions.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Load data and train model with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text('🔬 Analyzing skincare database...')
    progress_bar.progress(25)
    df = generate_skincare_dataset()
    
    status_text.text('🤖 Training AI recommendation engine...')
    progress_bar.progress(75)
    model, encoders, accuracy = train_model(df)
    
    status_text.text('✅ Ready to provide recommendations!')
    progress_bar.progress(100)
    
    # Clear progress indicators
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Features section
        st.markdown('<h2 class="section-header">🚀 Why Choose DermaCare AI?</h2>', unsafe_allow_html=True)
        
        features_col1, features_col2 = st.columns(2)
        
        with features_col1:
            st.markdown('''
            <div class="feature-card">
                <h3>🎯 Personalized Analysis</h3>
                <p>Our AI analyzes your unique skin profile considering type, age, concerns, climate, and lifestyle factors.</p>
            </div>
            
            <div class="feature-card">
                <h3>🏆 Science-Backed</h3>
                <p>Recommendations based on dermatological research and 10,000+ skin profiles for maximum effectiveness.</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with features_col2:
            st.markdown('''
            <div class="feature-card">
                <h3>💰 Budget-Friendly</h3>
                <p>Find effective products within your budget, from affordable drugstore to luxury skincare options.</p>
            </div>
            
            <div class="feature-card">
                <h3>📱 Easy to Follow</h3>
                <p>Step-by-step routines with clear instructions and product recommendations for your lifestyle.</p>
            </div>
            ''', unsafe_allow_html=True)
    
    with col2:
        # Quick stats
        st.markdown('''
        <div class="profile-card">
            <h3 style="text-align: center; margin-bottom: 1.5rem;">📊 Platform Stats</h3>
        </div>
        ''', unsafe_allow_html=True)
        
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.markdown('''
            <div class="metric-card">
                <h2 style="color: #667eea; margin: 0;">10K+</h2>
                <p style="margin: 0.5rem 0 0 0;">Skin Profiles</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with stats_col2:
            st.markdown(f'''
            <div class="metric-card">
                <h2 style="color: #667eea; margin: 0;">{accuracy:.0%}</h2>
                <p style="margin: 0.5rem 0 0 0;">AI Accuracy</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Sidebar for user input with enhanced styling
    st.sidebar.markdown("## 🔍 Skin Analysis Questionnaire")
    st.sidebar.markdown("*Help us understand your skin better*")
    
    skin_type = st.sidebar.selectbox(
        "🧴 What's your skin type?",
        ['Oily', 'Dry', 'Combination', 'Sensitive', 'Normal'],
        help="Not sure? Oily skin feels greasy, dry feels tight, combination has both, sensitive reacts easily."
    )
    
    age_group = st.sidebar.selectbox(
        "📅 What's your age group?",
        ['Teens (13-19)', 'Young Adult (20-29)', 'Adult (30-39)', 'Mature (40+)'],
        help="Age affects skin needs - younger skin may focus on acne, mature skin on anti-aging."
    )
    
    primary_concern = st.sidebar.selectbox(
        "🎯 What's your main skin concern?",
        ['Acne & Breakouts', 'Fine Lines & Wrinkles', 'Dark Spots & Hyperpigmentation', 
         'Dryness & Dehydration', 'Sensitivity & Redness', 'Excess Oil & Large Pores', 
         'Dullness & Uneven Texture', 'Loss of Firmness'],
        help="Choose your most pressing concern - we'll prioritize products to address this."
    )
    
    climate = st.sidebar.selectbox(
        "🌤️ What's your climate like?",
        ['Humid & Hot', 'Dry & Arid', 'Moderate & Temperate', 'Cold & Windy'],
        help="Climate affects skin hydration needs and product absorption."
    )
    
    lifestyle = st.sidebar.selectbox(
        "🏃‍♀️ How active is your lifestyle?",
        ['Very Active (Exercise 5+ times/week)', 'Moderately Active (Exercise 2-4 times/week)', 
         'Lightly Active (Exercise 1-2 times/week)', 'Sedentary (Minimal exercise)'],
        help="Active lifestyles may need oil control and sweat-resistant products."
    )
    
    budget = st.sidebar.selectbox(
        "💳 What's your budget preference?",
        ['Budget-Friendly ($5-$20)', 'Mid-Range ($20-$60)', 'Luxury ($60+)'],
        help="We'll recommend products within your price range without compromising quality."
    )
    
    # Enhanced recommendation button (removed unsupported `type="primary"`)
    if st.sidebar.button("✨ Get My Skincare Recommendations"):
        with st.spinner('🔮 Analyzing your skin profile and generating recommendations...'):
            time.sleep(1)  # Add slight delay for better UX
            
            # Make prediction
            user_input = pd.DataFrame({
                'skin_type': [encoders['skin_type'].transform([skin_type])[0]],
                'age_group': [encoders['age_group'].transform([age_group])[0]],
                'primary_concern': [encoders['primary_concern'].transform([primary_concern])[0]],
                'climate': [encoders['climate'].transform([climate])[0]],
                'lifestyle': [encoders['lifestyle'].transform([lifestyle])[0]],
                'budget': [encoders['budget'].transform([budget])[0]]
            })
            
            prediction = model.predict(user_input)[0]
            recommended_routine = encoders['routine'].inverse_transform([prediction])[0]
            
            # Get routine details
            routine_info = df[df['recommended_routine'] == recommended_routine].iloc[0]
            
            # Display enhanced recommendations
            st.markdown('<h2 class="section-header">🎯 Your Personalized Skincare Plan</h2>', unsafe_allow_html=True)
            
            recommendation_col1, recommendation_col2 = st.columns([3, 1])
            
            with recommendation_col1:
                st.markdown(f'''
                <div class="recommendation-card">
                    <h2 style="margin-bottom: 1rem;">🌟 {recommended_routine}</h2>
                    <p style="font-size: 1.1rem; margin-bottom: 1rem;"><strong>Perfect for:</strong> {skin_type} skin with {primary_concern.lower()} concerns in {climate.lower()} climates</p>
                    <p style="font-style: italic;">{routine_info['routine_description']}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Enhanced routine steps
                st.markdown('### 📋 Your Complete Routine')
                
                routine_steps = routine_info['routine_steps'].split(' → ')
                for i, step in enumerate(routine_steps, 1):
                    if i <= 2:  # Morning routine
                        time_label = "🌅 Morning"
                    elif i <= 4:  # Evening routine  
                        time_label = "🌙 Evening"
                    else:  # Additional steps
                        time_label = "✨ As needed"
                    
                    st.markdown(f'''
                    <div class="step-card">
                        <h4 style="margin: 0 0 0.5rem 0;">{time_label} - Step {i}</h4>
                        <h3 style="margin: 0;">{step}</h3>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Brand recommendations with enhanced styling
                st.markdown('### 🏷️ Recommended Brands')
                st.markdown(f'*{routine_info["brand_description"]}*')
                
                brands = routine_info['recommended_brands'].split(', ')
                brand_cols = st.columns(len(brands))
                
                for i, brand in enumerate(brands):
                    with brand_cols[i]:
                        st.markdown(f'''
                        <div class="brand-card">
                            <h4 style="margin: 0 0 0.5rem 0;">{brand}</h4>
                            <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Trusted Quality</p>
                        </div>
                        ''', unsafe_allow_html=True)
            
            with recommendation_col2:
                # Enhanced user profile
                st.markdown('''
                <div class="profile-card">
                    <h3 style="text-align: center; margin-bottom: 1.5rem;">👤 Your Profile</h3>
                </div>
                ''', unsafe_allow_html=True)
                
                profile_data = {
                    "🧴 Skin Type": skin_type,
                    "📅 Age Group": age_group,
                    "🎯 Main Concern": primary_concern,
                    "🌤️ Climate": climate,
                    "🏃‍♀️ Lifestyle": lifestyle,
                    "💳 Budget": budget
                }
                
                for label, value in profile_data.items():
                    st.markdown(f"**{label}:** {value}")
                
                # Model confidence
                st.markdown("---")
                st.markdown("### 🤖 AI Confidence")
                confidence = accuracy * 100
                st.progress(confidence/100)
                st.write(f"**{confidence:.1f}%** Match Accuracy")
    
    # Skincare education section
    st.markdown('<h2 class="section-header">📚 Skincare Education</h2>', unsafe_allow_html=True)
    
    tips = get_skincare_tips()
    tip_cols = st.columns(len(tips))
    
    for i, tip in enumerate(tips):
        with tip_cols[i]:
            st.markdown(f'''
            <div class="tip-box">
                <h4 style="margin: 0 0 0.5rem 0;">{tip['title']}</h4>
                <p style="margin: 0; font-size: 0.9rem;">{tip['content']}</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Enhanced analytics dashboard
    st.markdown('<h2 class="section-header">📊 Skincare Insights Dashboard</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Skin Analysis", "🎯 Concerns Breakdown", "💰 Budget Trends", "🌍 Climate Impact"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.pie(df, names='skin_type', title='Global Skin Type Distribution', 
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig1.update_layout(font=dict(size=14))
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            skin_routine = df.groupby(['skin_type', 'recommended_routine']).size().reset_index(name='count')
            fig2 = px.bar(skin_routine, x='skin_type', y='count', color='recommended_routine',
                         title='Recommended Routines by Skin Type',
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig2.update_layout(font=dict(size=12))
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.histogram(df, x='primary_concern', title='Most Common Skin Concerns',
                              color_discrete_sequence=['#667eea'])
            fig3.update_xaxis(tickangle=45)
            fig3.update_layout(font=dict(size=12))
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            concern_age = df.groupby(['primary_concern', 'age_group']).size().reset_index(name='count')
            # sunburst expects hierarchical counts; pivot to get counts per age_group->primary_concern
            fig4 = px.sunburst(concern_age, path=['age_group', 'primary_concern'], values='count',
                              title='Concerns by Age Group')
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig5 = px.histogram(df, x='budget', title='Budget Preferences',
                              color_discrete_sequence=['#764ba2'])
            st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            # Percentage of budgets within each age group
            budget_age = pd.crosstab(df['age_group'], df['budget'], normalize='index') * 100
            budget_age_reset = budget_age.reset_index()
            # build stacked bar showing budget distribution per age group
            fig6 = px.bar(budget_age_reset, x='age_group', y=budget_age.columns.tolist(),
                          title='Budget Distribution by Age Group', barmode='stack')
            fig6.update_layout(xaxis_title='Age Group', yaxis_title='Percentage (%)')
            st.plotly_chart(fig6, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            climate_counts = df['climate'].value_counts().reset_index()
            climate_counts.columns = ['climate', 'count']
            fig7 = px.bar(climate_counts, x='climate', y='count', title='Profiles by Climate')
            st.plotly_chart(fig7, use_container_width=True)
        
        with col2:
            # Impact of climate on dryness/dehydration concerns
            dry_concerns = df[df['primary_concern'].str.contains('Dryness|Dehydration', regex=True)]
            if not dry_concerns.empty:
                climate_dry = dry_concerns['climate'].value_counts().reset_index()
                climate_dry.columns = ['climate', 'count']
                fig8 = px.pie(climate_dry, names='climate', values='count', title='Dryness Concerns by Climate')
                st.plotly_chart(fig8, use_container_width=True)
            else:
                st.info("No dryness/dehydration concerns found in the dataset sample.")
    
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ — DermaCare AI | Data simulated for demo purposes only.")

if __name__ == "__main__":
    main()
