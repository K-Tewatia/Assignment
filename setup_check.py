# setup_check.py
import os
from dotenv import load_dotenv

def check_setup():
    """Check if the application is properly configured"""
    
    print("=" * 60)
    print("Marketing ROI Optimizer - Setup Check")
    print("=" * 60)
    print()
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        print()
        print("Creating a template .env file...")
        create_template_env()
        print("✅ Template .env file created!")
        print()
        print("Please edit .env file and add your API keys, then run this script again.")
        return False
    
    # Try to load with different encodings
    loaded = False
    encodings = ['utf-8', 'utf-16', 'utf-8-sig', 'latin-1']
    
    for encoding in encodings:
        try:
            # Try to read the file with this encoding
            with open('.env', 'r', encoding=encoding) as f:
                content = f.read()
            
            # If successful, rewrite with UTF-8
            if encoding != 'utf-8':
                print(f"⚠️  .env file was encoded as {encoding}, converting to UTF-8...")
                with open('.env', 'w', encoding='utf-8') as f:
                    f.write(content)
                print("✅ Converted to UTF-8")
            
            # Load environment variables
            load_dotenv(encoding='utf-8')
            loaded = True
            print("✅ .env file found and loaded")
            break
            
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with encoding {encoding}: {e}")
            continue
    
    if not loaded:
        print("❌ Could not read .env file with any encoding")
        print("   Recreating .env file...")
        create_template_env()
        print("✅ New .env file created")
        print("   Please add your API keys and run this script again.")
        return False
    
    print()
    
    # Check Pinecone API key
    pinecone_key = os.getenv('PINECONE_API_KEY')
    if pinecone_key and pinecone_key != 'your_pinecone_key_here':
        print(f"✅ PINECONE_API_KEY found ({pinecone_key[:10]}...)")
    else:
        print("❌ PINECONE_API_KEY not set or is placeholder")
        print("   Please edit .env file and add your actual Pinecone API key")
        print("   Get it from: https://www.pinecone.io/")
        return False
    
    # Check Mistral API key
    mistral_key = os.getenv('MISTRAL_API_KEY')
    if mistral_key and mistral_key != 'your_mistral_key_here':
        print(f"✅ MISTRAL_API_KEY found ({mistral_key[:10]}...)")
    else:
        print("❌ MISTRAL_API_KEY not set or is placeholder")
        print("   Please edit .env file and add your actual Mistral API key")
        print("   Get it from: https://console.mistral.ai/")
        return False
    
    print()
    
    # Check optional settings
    index_name = os.getenv('PINECONE_INDEX_NAME', 'marketing-roi-optimizer')
    print(f"ℹ️  Pinecone Index: {index_name}")
    
    model_name = os.getenv('MISTRAL_MODEL', 'mistral-large-latest')
    print(f"ℹ️  Mistral Model: {model_name}")
    
    print()
    print("=" * 60)
    print("✅ Setup check complete! You're ready to run the app.")
    print("=" * 60)
    print()
    print("Run: streamlit run app.py")
    
    return True

def create_template_env():
    """Create a template .env file with UTF-8 encoding"""
    template = """# Marketing ROI Optimizer Configuration
# Replace the placeholder values with your actual API keys

# Pinecone API Key (Required)
# Get it from: https://www.pinecone.io/
PINECONE_API_KEY=your_pinecone_key_here

# Mistral API Key (Required)
# Get it from: https://console.mistral.ai/
MISTRAL_API_KEY=your_mistral_key_here

# Optional: Pinecone Index Name (Default: marketing-roi-optimizer)
PINECONE_INDEX_NAME=marketing-roi-optimizer

# Optional: Mistral Model (Default: mistral-large-latest)
# Options: mistral-large-latest, mistral-medium, mistral-small
MISTRAL_MODEL=mistral-large-latest
"""
    
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(template)

if __name__ == "__main__":
    check_setup()
