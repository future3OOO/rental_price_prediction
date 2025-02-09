import os
import sys

def main():
    try:
        from rental_data_script import RentalDataAnalysis
    except ImportError as e:
        print(f"Error importing RentalDataAnalysis: {e}")
        print(f"Python version: {sys.version}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir()}")
        return

    try:
        # Use the correct path to your CSV file
        data_path = r'C:\Users\Property Partner\Desktop\propert partner\Dev work\Rental project\cleaned_rental_data.csv'
        
        # Create an instance of RentalDataAnalysis
        analysis = RentalDataAnalysis(data_path)
        
        # Run the full analysis
        analysis.run_full_analysis()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()