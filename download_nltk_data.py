import nltk
import ssl
import os

# Ensure NLTK data is downloaded to a persistent and accessible location
# On Render, /opt/render/project/src/nltk_data is generally a good place
# You can set NLTK_DATA environment variable if needed, but the default often works
# if it's placed relative to your project root during build.
nltk_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path) # Add to NLTK's search path

# Handle potential SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK data
print("Attempting to download NLTK 'punkt_tab'...")
try:
    nltk.download('punkt_tab', download_dir=nltk_data_path)
except Exception as e:
    print(f"Error downloading punkt_tab: {e}")

print("Attempting to download NLTK 'stopwords'...")
try:
    nltk.download('stopwords', download_dir=nltk_data_path)
except Exception as e:
    print(f"Error downloading stopwords: {e}")

print("Attempting to download NLTK 'punkt'...")
try:
    nltk.download('punkt', download_dir=nltk_data_path)
except Exception as e:
    print(f"Error downloading punkt: {e}")

print("All required NLTK data download commands initiated.")