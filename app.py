# import base64
# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score

# # Set page config (must be first command)
# st.set_page_config(page_title="🚘 Car Price Predictor", layout="wide")

# # 👉 Function to add background image from local file
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image:
#         encoded = base64.b64encode(image.read()).decode()
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url("data:image/jpeg;base64,{encoded}");
#             background-size: cover;
#             background-position: center;
#             background-attachment: fixed;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # Call function to set background image
# add_bg_from_local("car img.jpeg")  # <-- make sure this file is in the same folder

# # Make sidebar background transparent with blur
# st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"] {
#         background: rgba(255, 255, 255, 0);
#         backdrop-filter: blur(10px);
#         -webkit-backdrop-filter: blur(10px);
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Title and header with emoji
# st.title("🚗 Car Price Prediction with Machine Learning")
# st.markdown("### Predict the selling price of your car with ease!")

# # Load Data
# df = pd.read_csv("car data.csv")

# # Select relevant columns
# features = ['Year', 'Driven_kms', 'Present_Price']
# target = 'Selling_Price'
# df = df[features + [target, 'Car_Name']].dropna()

# # Prepare features and target arrays
# X = df[features]
# y = df[target]

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predictions on test data
# y_pred_test = model.predict(X_test)
# score = r2_score(y_test, y_pred_test)

# # Sidebar inputs
# st.sidebar.header("🛠️ Customize Your Car Details")

# # Car Name dropdown
# car_names = df['Car_Name'].unique()
# car_name = st.sidebar.selectbox("Select Car Name", options=car_names)

# # Year slider
# year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
# year = st.sidebar.slider("Year of Manufacture", year_min, year_max, year_max)

# # Driven Kms slider
# driven_min, driven_max = int(df['Driven_kms'].min()), int(df['Driven_kms'].max())
# driven_kms = st.sidebar.slider("Driven Kilometers", driven_min, driven_max, int(driven_max/2), step=1000)

# # Present price slider
# present_price_min, present_price_max = float(df['Present_Price'].min()), float(df['Present_Price'].max())
# present_price = st.sidebar.slider("Present Price (in lakhs)", present_price_min, present_price_max, float(present_price_max/2))

# # Prediction input dataframe
# input_df = pd.DataFrame([[year, driven_kms, present_price]], columns=features)

# # Predict
# predicted_price = model.predict(input_df)[0]

# # Display prediction & model accuracy using columns
# col1, col2 = st.columns(2)
# col1.metric(label="💰 Predicted Selling Price", value=f"₹ {predicted_price:,.2f} lakhs")
# col2.metric(label="📊 Model Accuracy (R² Score)", value=f"{score:.3f}")

# st.markdown(f"### Prediction for: **{car_name}**")

# # Plot Actual vs Predicted with transparent background
# st.subheader("📉 Model Performance: Actual vs Predicted Selling Prices")

# sns.set_style("darkgrid")
# fig, ax = plt.subplots(figsize=(10,6), facecolor='none')  # transparent figure background

# sns.scatterplot(x=y_test, y=y_pred_test, ax=ax, alpha=0.7, edgecolor='k', label='Test Data')

# lims = [
#     min(y_test.min(), y_pred_test.min(), predicted_price),
#     max(y_test.max(), y_pred_test.max(), predicted_price)
# ]
# ax.plot(lims, lims, 'r--', label='Perfect Prediction (y=x)')
# ax.set_xlim(lims)
# ax.set_ylim(lims)

# # Highlight user's predicted point
# ax.scatter(predicted_price, predicted_price, color='red', s=150, label='Your Car Prediction', marker='X')

# ax.set_xlabel("Actual Selling Price (in lakhs)", fontsize=14)
# ax.set_ylabel("Predicted Selling Price (in lakhs)", fontsize=14)
# ax.legend(fontsize=12)
# ax.grid(True)

# # Make plot background transparent
# ax.patch.set_alpha(0)  # axes background transparent
# fig.patch.set_alpha(0)  # figure background transparent

# st.pyplot(fig)








































# import streamlit as st
# import base64
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score

# # 1️⃣ Must be FIRST Streamlit command
# st.set_page_config(page_title="🚘 Car Price Predictor", layout="wide")

# # Function to add background image and style sidebar transparent
# def add_bg_and_sidebar_style(image_file):
#     with open(image_file, "rb") as image:
#         encoded = base64.b64encode(image.read()).decode()

#     st.markdown(
#         f"""
#         <style>
#         /* Main app background */
#         .stApp {{
#             background-image: url("data:image/jpeg;base64,{encoded}");
#             background-size: cover;
#             background-position: center;
#             background-attachment: fixed;
#             # color: white;
#         }}

#         /* Sidebar transparent */
#         [data-testid="stSidebar"] > div:first-child {{
#             background-color: transparent !important;
#             # box-shadow: none !important;
#         }}

#         /* Sidebar content text white */
#         [data-testid="stSidebar"] * {{
#             # color: white !important;
#         }}

#         /* Widget labels white */
#         label, .css-10trblm {{
#             # color: white !important;
#         }}

#         /* Sidebar inputs transparent background */
#         [data-testid="stSidebar"] input, 
#         [data-testid="stSidebar"] select, 
#         [data-testid="stSidebar"] textarea {{
#             background-color: rgba(255, 255, 255, 0) !important;
#             # color: white !important;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )
# # Call the function early in the app
# add_bg_and_sidebar_style("car img.jpeg")



# # # 3️⃣ Add background
# # add_bg_from_local("car img.jpeg")

# # 4️⃣ Load data
# df = pd.read_csv("car data.csv")

# # 5️⃣ Select features and target
# features = ['Year', 'Driven_kms', 'Present_Price']
# target = 'Selling_Price'
# df = df[features + [target, 'Car_Name']].dropna()

# X = df[features]
# y = df[target]

# # 6️⃣ Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 7️⃣ Train model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # 8️⃣ Predictions and model score
# y_pred_test = model.predict(X_test)
# score = r2_score(y_test, y_pred_test)

# # 9️⃣ Sidebar inputs
# st.sidebar.header("🛠️ Customize Your Car Details")

# car_names = df['Car_Name'].unique()
# car_name = st.sidebar.selectbox("Select Car Name", options=car_names)

# year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
# year = st.sidebar.slider("Year of Manufacture", year_min, year_max, year_max)

# driven_min, driven_max = int(df['Driven_kms'].min()), int(df['Driven_kms'].max())
# driven_kms = st.sidebar.slider("Driven Kilometers", driven_min, driven_max, int(driven_max/2), step=1000)

# present_price_min, present_price_max = float(df['Present_Price'].min()), float(df['Present_Price'].max())
# present_price = st.sidebar.slider("Present Price (in lakhs)", present_price_min, present_price_max, float(present_price_max/2))

# # 10️⃣ Make prediction for user input
# input_df = pd.DataFrame([[year, driven_kms, present_price]], columns=features)
# predicted_price = model.predict(input_df)[0]

# # 11️⃣ Display prediction and model accuracy
# col1, col2 = st.columns(2)
# col1.metric(label="💰 Predicted Selling Price", value=f"₹ {predicted_price:,.2f} lakhs")
# col2.metric(label="📊 Model Accuracy (R² Score)", value=f"{score:.3f}")

# st.markdown(f"<h3 style='color:white;'>Prediction for: <strong>{car_name}</strong></h3>", unsafe_allow_html=True)

# # 12️⃣ Plot Actual vs Predicted
# st.subheader("📉 Model Performance: Actual vs Predicted Selling Prices")

# sns.set_style("darkgrid")
# fig, ax = plt.subplots(figsize=(10, 6))

# sns.scatterplot(x=y_test, y=y_pred_test, ax=ax, alpha=0.5, edgecolor='k', label='Test Data')

# lims = [
#     min(y_test.min(), y_pred_test.min(), predicted_price),
#     max(y_test.max(), y_pred_test.max(), predicted_price)
# ]

# ax.plot(lims, lims, 'r--', label='Perfect Prediction (y=x)')
# ax.set_xlim(lims)
# ax.set_ylim(lims)

# # Highlight user's predicted point (make marker bigger and red)
# ax.scatter(predicted_price, predicted_price, color='red', s=150, label='Your Car Prediction', marker='X')

# ax.set_xlabel("Actual Selling Price (in lakhs)", fontsize=14, color='white')
# ax.set_ylabel("Predicted Selling Price (in lakhs)", fontsize=14, color='white')
# ax.legend(fontsize=12)
# ax.grid(True)

# # Set background and tick colors of plot to transparent/white for better visibility
# ax.set_facecolor("none")
# ax.figure.patch.set_alpha(0)
# ax.tick_params(axis='x', colors='white')
# ax.tick_params(axis='y', colors='white')
# ax.spines['bottom'].set_color('white')
# ax.spines['left'].set_color('white')

# st.pyplot(fig)



























































import streamlit as st
import base64
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1️⃣ Set page config
st.set_page_config(page_title="🚘 Car Price Predictor", layout="wide")

# 2️⃣ Add background and sidebar styling
def add_bg_and_sidebar_style(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        /* Main app background */
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        [data-testid="stSidebar"] > div:first-child {{
            background-color: rgba(0, 0, 0, 0.25) !important;
            box-shadow: none !important;
        }}

        [data-testid="stSidebar"] * {{
            color: black !important;
        }}

        .main, .main * {{
            color: white !important;
        }}

        [data-testid="stSidebar"] input, 
        [data-testid="stSidebar"] select, 
        [data-testid="stSidebar"] textarea {{
            background-color: rgba(255, 255, 255, 0.75) !important;
            color: white !important;
            border: 1px solid white !important;
        }}

        .css-1cpxqw2, .css-1x8cf1d, .css-1d391kg {{
            color: white !important;
        }}

        .element-container .stMetric label {{
            color: white !important;
        }}
        
        
        </style>
        """,
        unsafe_allow_html=True,
    )

# Then call it:
add_bg_and_sidebar_style("car img.jpeg")
# 3️⃣ Call background + style setup
add_bg_and_sidebar_style("car img.jpeg")  # Replace with your image path

# 4️⃣ Load and prepare data
df = pd.read_csv("car data.csv")

features = ['Year', 'Driven_kms', 'Present_Price']
target = 'Selling_Price'
df = df[features + [target, 'Car_Name']].dropna()

X = df[features]
y = df[target]

# 5️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Model training
model = LinearRegression()
model.fit(X_train, y_train)

# 7️⃣ Prediction and accuracy
y_pred_test = model.predict(X_test)
score = r2_score(y_test, y_pred_test)

# 8️⃣ Sidebar - user input
st.sidebar.header("🛠️ Customize Your Car Details")

car_names = df['Car_Name'].unique()
car_name = st.sidebar.selectbox("Select Car Name", options=car_names)

year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
year = st.sidebar.slider("Year of Manufacture", year_min, year_max, year_max)

driven_min, driven_max = int(df['Driven_kms'].min()), int(df['Driven_kms'].max())
driven_kms = st.sidebar.slider("Driven Kilometers", driven_min, driven_max, int(driven_max/2), step=1000)

present_price_min, present_price_max = float(df['Present_Price'].min()), float(df['Present_Price'].max())
present_price = st.sidebar.slider("Present Price (in lakhs)", present_price_min, present_price_max, float(present_price_max/2))

# 9️⃣ Make prediction
input_df = pd.DataFrame([[year, driven_kms, present_price]], columns=features)
predicted_price = model.predict(input_df)[0]

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div style="background-color: rgba(0,0,0,0.4); padding: 20px 25px; border-radius: 20px; display: inline-block;">
            <p style="color: white; font-size: 30px; margin: 1;">💰 Predicted Selling Price</p>
            <p style="color: white; font-size: 25px; font-weight: bold; margin: 1;">₹ {predicted_price:,.2f} lakhs</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div style="background-color: rgba(0,0,0,0.4); padding: 20px 25px; border-radius: 20px; display: inline-block;">
            <p style="color: white; font-size: 30px; margin: 1;">📊 Model Accuracy (R² Score)</p>
            <p style="color: white; font-size: 25px; font-weight: bold; margin: 1;">{score:.3f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 🔢 Plot actual vs predicted
st.markdown(
    """
    <h2 style='color: white;'>📉 Model Performance: Actual vs Predicted Selling Prices</h2>
    """,
    unsafe_allow_html=True,
)

sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(x=y_test, y=y_pred_test, ax=ax, alpha=0.5, edgecolor='k', label='Test Data')

lims = [
    min(y_test.min(), y_pred_test.min(), predicted_price),
    max(y_test.max(), y_pred_test.max(), predicted_price)
]

ax.plot(lims, lims, 'r--', label='Perfect Prediction (y=x)')
ax.set_xlim(lims)
ax.set_ylim(lims)

# User's prediction point
ax.scatter(predicted_price, predicted_price, color='red', s=150, label='Your Car Prediction', marker='X')

# Axes and text color
ax.set_xlabel("Actual Selling Price (in lakhs)", fontsize=14, color='white')
ax.set_ylabel("Predicted Selling Price (in lakhs)", fontsize=14, color='white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.set_facecolor("none")
ax.figure.patch.set_alpha(0)
ax.legend(fontsize=12)

st.pyplot(fig)
