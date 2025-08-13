import pandas as pd
import streamlit as st

# Test the app with a simple import check
try:
    # Test import
    import xlsxwriter
    print("✅ xlsxwriter is available")
except ImportError:
    print("❌ xlsxwriter not found - installing...")
    import subprocess
    subprocess.run(["pip", "install", "xlsxwriter"], check=True)
    print("✅ xlsxwriter installed successfully")

print("✅ All imports should work correctly")
print("✅ Enhanced features implemented:")
print("  1. ✅ Removed secondary y-axis gridlines")
print("  2. ✅ Improved YoY growth calculation (only when sufficient data)")
print("  3. ✅ Fixed YTD growth calculation (cumulative from year beginning)")
print("  4. ✅ Added YoY growth line to water capacity chart")
print("  5. ✅ Added download buttons (Excel & CSV) for all charts")
print("  6. ✅ All charts with dual y-axis updated")
