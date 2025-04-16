# -*-
# coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog, filedialog, messagebox
from tkcalendar import DateEntry
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib
import re # <--- Make sure re is imported at the top of your script
import traceback

matplotlib.use("TkAgg")  # Use Tkinter backend for Matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import numpy as np
import sys
import pandas as pd
import time
import threading
import webbrowser # <-- Import webbrowser
import os       # <-- Import os

# For the LSTM model
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler # StandardScaler not used, commented out
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

# from tensorflow.keras.models import load_model # Keep for potential future model saving/loading

# Additional imports for sentiment analysis
try:
    from gdeltdoc import GdeltDoc, Filters

    HAS_GDELT = True
except ImportError:
    print("Warning: gdeltdoc library not found. Sentiment analysis will be disabled.")
    HAS_GDELT = False
try:
    from newspaper import Article, Config
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    HAS_NEWS_LIBS = True
except ImportError:
    print("Warning: newspaper3k or vaderSentiment not found. Sentiment analysis will be disabled.")
    HAS_NEWS_LIBS = False

from sklearn.linear_model import LinearRegression

# Disable sentiment features if libraries are missing
PERFORM_SENTIMENT = HAS_GDELT and HAS_NEWS_LIBS
FEEDBACK_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSeWubehsK_20yMWkj8VpaoDcQA6cSmsKDS07fSS91W2lmbK7A/viewform?usp=dialog" # <-- *** REPLACE WITH YOUR ACTUAL URL ***
DOCUMENTATION_FILENAME = "Stock Analysis PC.pdf"

# ---- Set up Newspaper3k config with a custom user-agent ----
if HAS_NEWS_LIBS:
    newspaper_user_agent = (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/115.0.0.0 Safari/537.36'
    )
    newspaper_config = Config()
    newspaper_config.browser_user_agent = newspaper_user_agent
    newspaper_config.request_timeout = 15  # Add timeout
    newspaper_config.fetch_images = False  # Don't need images
    newspaper_config.memoize_articles = False  # Disable caching if not needed across runs

# --- Helper Function for Resource Path (Dev vs. Bundle) ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        # Adjust based on where the file is placed by 'datas' in spec
        # If datas=('Stock Analysis PC.pdf', '.') -> relative path starts from base_path
        # If datas=('docs/Stock Analysis PC.pdf', 'docs') -> relative_path should be 'docs/Stock Analysis PC.pdf'
        base_path = sys._MEIPASS
    except Exception:
        # _MEIPASS not set, running in normal Python environment
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def get_stock_data(symbol, start_date, end_date):
    """
    Try to download stock data using yfinance.
    Returns an empty DataFrame if symbol is invalid, data is unavailable,
    or another error occurs during download/processing.
    """
    end_date_yf = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    data = pd.DataFrame() # Initialize empty

    # Check for obviously invalid symbol patterns early (optional)
    # if not symbol or not isinstance(symbol, str): # Basic check
    #    print(f"Invalid symbol type provided: {symbol}")
    #    return pd.DataFrame()

    try:
        print(f"Fetching yfinance data for {symbol} from {start_date} to {end_date_yf.strftime('%Y-%m-%d')}")

        # --- Wrap yf.download() to catch internal yfinance errors ---
        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date_yf.strftime('%Y-%m-%d'),
                auto_adjust=True,
                progress=False,
                timeout=15 # Keep or adjust timeout
            )
            # Explicitly check if yfinance returned None (very rare, but defensive)
            if data is None:
                 print(f"Warning: yf.download returned None unexpectedly for {symbol}")
                 data = pd.DataFrame() # Treat as empty

        except Exception as yf_download_err:
            # This catches crashes *during* the download attempt itself
            print(f"ERROR during yfinance download call for '{symbol}': {type(yf_download_err).__name__} - {yf_download_err}")
            # traceback.print_exc() # Optional: Uncomment for full traceback in console
            data = pd.DataFrame() # Ensure data is empty on exception

        # --- Process data ONLY if download returned something ---
        if data.empty:
            # This message covers both explicit empty return and caught exceptions
            print(f"yf.download completed but resulted in empty data for {symbol}.")
        else:
            # --- Filtering and Column Processing ---
            # Filter end date
            data = data[data.index <= pd.to_datetime(end_date)]
            if data.empty:
                 print(f"Data became empty after filtering end date for {symbol}.")
                 return pd.DataFrame() # Return empty if filtering makes it empty

            print(f"yfinance data fetched successfully. Shape before processing columns: {data.shape}")

            # Column Processing (Keep as before)
            # ... (MultiIndex check, lowercasing, required cols check, index conversion) ...
            try:
                if isinstance(data.columns, pd.MultiIndex):
                     data.columns = [col[0].lower() for col in data.columns.values]
                elif isinstance(data.columns, pd.Index):
                     data.columns = [col.lower() for col in data.columns]
                else: pass # Warning already printed if needed

                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    print(f"Core columns missing after processing: {missing_cols}. Returning empty.")
                    return pd.DataFrame()

                data.index = pd.to_datetime(data.index)
                print(f"Processed columns: {data.columns}")

            except Exception as processing_err:
                 print(f"ERROR during column/index processing for {symbol}: {processing_err}")
                 traceback.print_exc()
                 return pd.DataFrame() # Return empty on processing error

    except Exception as outer_err:
        # Catch any other unexpected errors
        print(f"Unexpected error in get_stock_data for {symbol}: {outer_err}")
        traceback.print_exc()
        return pd.DataFrame() # Ensure empty return

    # --- Final Log ---
    if data.empty and symbol:
        # This log confirms that after everything, we are returning empty
        print(f"Final check: No valid data retrieved for symbol '{symbol}'. Returning empty DataFrame.")

    return data

# --- End of Modified Column Processing ---

def compute_rsi(series, period=14):
    """Compute the RSI for a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use Exponential Moving Average (EMA) for RSI calculation (common practice)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)  # Replace 0 loss with NaN to avoid infinite RS
    rsi = 100 - (100 / (1 + rs))

    # Handle cases where avg_loss was 0 (RSI should be 100)
    rsi = rsi.fillna(100)  # If rs is NaN because avg_loss was 0, RSI is 100
    # Handle initial NaNs
    # rsi = rsi.fillna(method='bfill') # Backfill initial NaNs if needed
    rsi = rsi.bfill()  # Use recommended method

    return rsi


class StockAnalysisApp:
    def __init__(self):
        print("DEBUG: Initializing StockAnalysisApp...") # Add debug print
        self.root = tk.Tk()
        # self.root.withdraw()  # --- COMMENT OUT or REMOVE this line ---
        print("DEBUG: Root window created (NOT withdrawn yet).")
        # ... rest of your __init__ remains the same ...
        self.sentiment_data = {}
        self.article_urls = []
        self.plot_figure = None  # <-- ADD THIS
        self.plot_window = None
        self.original_xlim = None
        self.original_ylim = None
        self.annotation = None
        self.feedback_popup = None
        self.initial_dialog_shown = False
        self.loading_message_var = tk.StringVar()
        # Initialize the result variable here too
        self.initial_dialog_result = tk.StringVar(value="cancel")
        # --- MAKE ROOT INVISIBLE --- <<< ADD THESE LINES
        self.root.title("") # No title
        self.root.geometry("1x1-0-0") # Tiny size, off-screen top-left
        print("DEBUG: Root window created (Made nearly invisible).")
        self.plot_canvas_widget = None # Make sure this is initialized

    # --- REVISED Initial Dialog Function (Called BEFORE mainloop) ---
    def show_initial_dialog(self):
        """Shows the ONE-TIME introductory dialog. Blocks until closed."""
        print("DEBUG: Entered show_initial_dialog function.")
        if self.initial_dialog_shown:
             print("DEBUG: Initial dialog already shown, skipping.")
             return "ok" # Return ok if already shown

        dialog = None
        # Use the instance variable directly, default set in __init__
        result_captured = "cancel" # Local var to track button press

        # Define helper functions
        def on_ok():
            print("DEBUG: OK button clicked.")
            self.initial_dialog_result.set("ok")
            nonlocal result_captured; result_captured = "ok"
            if dialog and dialog.winfo_exists(): dialog.destroy()
        def on_docs():
            print("DEBUG: Documentation button clicked.")
            self.open_documentation()
            if dialog and dialog.winfo_exists(): dialog.lift()
        def on_cancel():
            print("DEBUG: Dialog closed via X or implicitly cancelled.")
            self.initial_dialog_result.set("cancel")
            nonlocal result_captured; result_captured = "cancel"
            if dialog and dialog.winfo_exists(): dialog.destroy()

        try:
            # No need to deiconify root as it's not withdrawn
            print("DEBUG: Creating Toplevel...")
            dialog = tk.Toplevel(self.root) # Parent is the visible root
            dialog.title("Welcome - Stock Analysis PC")
            dialog.resizable(False, False)
            dialog.transient(self.root) # Keep transient
            print("DEBUG: Toplevel created.")

            # --- Widgets (Frame, Label, Buttons) ---
            # (Keep the widget creation/packing code using tk.Button)
            print("DEBUG: Creating Frame...")
            frame = tk.Frame(dialog, padx=20, pady=15); frame.pack(expand=True, fill="both")
            intro_text = ("\t\t          **Stock Analyzer**\n\nVisualize stock data, analyze indicators, and view forecasted price using LSTM & news sentiment.\n\nPlease read the Documentation for important details & disclaimers. This application need internet connection in order to work, please take a note on this!")
            print("DEBUG: Creating Label...")
            label = tk.Label(frame, text=intro_text, justify=tk.LEFT, wraplength=360); label.pack(pady=(0, 15))
            print("DEBUG: Creating Button Frame...")
            button_frame = tk.Frame(frame); button_frame.pack(side="bottom", fill="x", pady=(5, 0))
            print("DEBUG: Creating Docs Button...")
            docs_btn = tk.Button(button_frame, text="Documentation", command=on_docs, width=16); docs_btn.pack(side="right", padx=(12, 0))
            print("DEBUG: Creating OK Button...")
            ok_btn = tk.Button(button_frame, text="   OK   ", command=on_ok, width=10); ok_btn.pack(side="right")
            print("DEBUG: Buttons packed.")

            # --- Final Dialog Setup ---
            dialog.protocol("WM_DELETE_WINDOW", on_cancel)
            dialog.grab_set() # Keep modal

            # --- Centering ---
            print("DEBUG: Updating idletasks for geometry...")
            dialog.update_idletasks() # Calculate size
            width = dialog.winfo_reqwidth(); height = dialog.winfo_reqheight()
            if width < 100 or height < 50: width, height = 400, 180
            screen_w = dialog.winfo_screenwidth(); screen_h = dialog.winfo_screenheight()
            x = (screen_w // 2) - (width // 2); y = (screen_h // 2) - (height // 2)
            dialog.geometry(f'{width}x{height}+{x}+{y}')
            print("DEBUG: Dialog geometry set.")

            # --- Focus ---
            dialog.focus_set()
            ok_btn.focus()

            # --- Force Update (Might still help ensure visibility before wait) ---
            print("DEBUG: Forcing DIALOG update before wait_window...")
            dialog.update_idletasks()
            dialog.update()
            print("DEBUG: Dialog update forced. Reaching wait_window...")

            # --- Use wait_window SINCE mainloop HASN'T started yet ---
            dialog.wait_window()
            print(f"DEBUG: wait_window finished. Captured Result: '{result_captured}'")

        except Exception as e:
            print(f"ERROR INSIDE show_initial_dialog: {type(e).__name__} - {e}")
            traceback.print_exc()
            result_captured = "cancel"; self.initial_dialog_result.set("cancel")
            if dialog and dialog.winfo_exists(): dialog.destroy()
        finally:
             print("DEBUG: Exiting show_initial_dialog function.")
             self.initial_dialog_shown = True # Mark as shown

        return result_captured # Return the captured local result

    # --- New Helper Function to Open Documentation ---
    def open_documentation(self):
        """Opens the bundled documentation PDF."""
        try:
            doc_path = resource_path(DOCUMENTATION_FILENAME)
            print(f"Attempting to open documentation at: {doc_path}")
            if os.path.exists(doc_path):
                 webbrowser.open(f'file:///{os.path.realpath(doc_path)}') # Use file URI
                 # webbrowser.open_new_tab(doc_path) # Alternative
            else:
                 print(f"ERROR: Documentation file not found at: {doc_path}")
                 messagebox.showerror("Error", f"Documentation file not found:\n{DOCUMENTATION_FILENAME}\nPlease ensure it was bundled correctly.", parent=self.root)
        except Exception as e:
            print(f"Error opening documentation: {e}")
            messagebox.showerror("Error", f"Could not open documentation:\n{e}", parent=self.root)

    # --- New Helper Function to Open Feedback Form ---
    def open_feedback_form(self):
        """Opens the Google Form URL in a new browser tab."""
        try:
            print(f"Opening feedback form: {FEEDBACK_FORM_URL}")
            webbrowser.open_new_tab(FEEDBACK_FORM_URL)
        except Exception as e:
            print(f"Error opening feedback form: {e}")
            messagebox.showerror("Error", f"Could not open feedback form link:\n{e}", parent=self.plot_window if self.plot_window and self.plot_window.winfo_exists() else self.root)

    # --- REVISED Helper Function (Position After Lift/Update) ---
    def show_feedback_popup(self):
        """Shows a temporary popup encouraging feedback - Attempt position after lift."""

        # Check if plot window exists (still important)
        if not self.plot_window or not self.plot_window.winfo_exists():
            print("DEBUG: Plot window doesn't exist, skipping feedback popup.")
            return
        # Check if popup already exists (still important)
        if self.feedback_popup and self.feedback_popup.winfo_exists():
            print("DEBUG: Feedback popup already exists, lifting.")
            self.feedback_popup.lift(); return

        print("DEBUG: Creating feedback popup.")
        self.feedback_popup = tk.Toplevel(self.plot_window)
        #self.feedback_popup.overrideredirect(True)
        self.feedback_popup.title("Feedback Welcome!") # <<<--- ADD a title
        self.feedback_popup.resizable(False, False)
        self.feedback_popup.transient(self.plot_window)

        # Styling
        popup_bg="#F0F0F0"; text_color="#333333"; font_family="Segoe UI"
        font_size_header=11; font_size_body=10
        self.feedback_popup.configure(bg=popup_bg)

        # Content Frame & Widgets (remain the same)
        frame = tk.Frame(self.feedback_popup, bg=popup_bg, padx=20, pady=15); frame.pack(expand=True, fill="both")
        msg_header = "Feedback Welcome!"
        header_label = ttk.Label(frame, text=msg_header, font=(font_family, font_size_header, "bold"), background=popup_bg, foreground=text_color, justify=tk.CENTER); header_label.pack(pady=(0, 5))
        msg_body = "Your input helps improve this tool.\nClick the 'Feedback' button below the plot."
        body_label = ttk.Label(frame, text=msg_body, font=(font_family, font_size_body), background=popup_bg, foreground=text_color, justify=tk.CENTER, wraplength=260); body_label.pack(pady=(0, 15))
        dismiss_button = ttk.Button(frame, text="Dismiss", command=self.close_feedback_popup, width=10); dismiss_button.pack(pady=(5, 0))
        print("DEBUG: Feedback popup widgets created.")

        # --- Positioning Logic (Revised Timing) ---
        try:
            # Step 1: Update popup to calculate its required size
            self.feedback_popup.update_idletasks()
            popup_w = self.feedback_popup.winfo_reqwidth() + 20
            popup_h = self.feedback_popup.winfo_reqheight() + 10
            print(f"DEBUG: Feedback popup required size: {popup_w}x{popup_h}")

            # Step 2: Get SCREEN dimensions
            screen_w = self.feedback_popup.winfo_screenwidth()
            screen_h = self.feedback_popup.winfo_screenheight()
            print(f"DEBUG: Screen dimensions: {screen_w}x{screen_h}")

            # Step 3: Calculate position for SCREEN centering
            x = (screen_w // 2) - (popup_w // 2)
            y = (screen_h // 2) - (popup_h // 2)
            print(f"DEBUG: Calculated popup position (screen center): X={x}, Y={y}")

            self.feedback_popup.update_idletasks()
            popup_w = self.feedback_popup.winfo_reqwidth() + 20;
            popup_h = self.feedback_popup.winfo_reqheight() + 10
            screen_w = self.feedback_popup.winfo_screenwidth();
            screen_h = self.feedback_popup.winfo_screenheight()
            x = (screen_w // 2) - (popup_w // 2);
            y = (screen_h // 2) - (popup_h // 2)
            self.feedback_popup.geometry(f'{popup_w}x{popup_h}+{x}+{y}')
            print(f"DEBUG: Setting feedback popup geometry to {popup_w}x{popup_h}+{x}+{y}")

        except Exception as geo_err:
            print(f"ERROR getting geometry or centering on screen: {geo_err}")
            self.feedback_popup.geometry('+300+300')

        # --- Schedule Auto-Close & Display ---
        # ---> ADD AUTO-CLOSE BACK <---
        self.feedback_popup.after(10000, self.close_feedback_popup)
        # ---------------------------
        self.feedback_popup.lift()
        #self.feedback_popup.grab_set()
        self.feedback_popup.update()
        print("DEBUG: Feedback popup displayed (will close in 10s).")

    # --- New Helper Function to Close Feedback Popup ---
    def close_feedback_popup(self):
        """Safely destroys the feedback popup window."""
        if self.feedback_popup and self.feedback_popup.winfo_exists():
            self.feedback_popup.destroy()
            self.feedback_popup = None

    def get_stock_symbol(self):
        """
        Gets stock symbol. Handles common formats including adding '.'
        before likely exchange suffixes (e.g., 1155KL -> 1155.KL, 0700 HK -> 0700.HK).
        Prioritizes formats already containing a dot.
        """
        # Use a loop to re-prompt if input is empty after stripping
        while True:
            symbol_input = simpledialog.askstring(
                "Input",
                "Enter the stock symbol (e.g., AAPL, MSFT, BRK-B, 1155.KL, 0700HK, 1802 TW):",
                parent=self.root
            )

            if symbol_input is None:
                # User clicked Cancel
                print("Symbol entry cancelled by user.")
                return None

            # Basic cleaning: remove leading/trailing spaces, uppercase
            symbol = symbol_input.strip().upper()

            if symbol:
                # Input is not empty after stripping, break the loop
                break
            else:
                # Input was empty or only whitespace
                messagebox.showwarning("Input Required", "Stock symbol cannot be empty. Please try again.", parent=self.root)
                # Loop continues to re-prompt

        # --- Normalization Strategy ---
        # 1. If it contains '.', assume user knows the yfinance format.
        # 2. Check for pattern: Digits followed immediately by Letters (likely suffix).
        # 3. Check for pattern: Digits followed by Space(s) followed by Letters.
        # 4. Otherwise, assume standard ticker or other format yfinance might handle.

        if '.' in symbol:
            print(f"Symbol contains '.', using as entered: {symbol}")
            return symbol

        # Check for Digits+Letters pattern (e.g., 1155KL)
        match_no_space = re.match(r"^(\d+)([A-Z]+)$", symbol)
        if match_no_space:
            number_part = match_no_space.group(1)
            letter_part = match_no_space.group(2)
            normalized_symbol = f"{number_part}.{letter_part}"
            print(f"Detected Number+Letters pattern, normalized to: {normalized_symbol}")
            return normalized_symbol

        # Check for Digits+Space(s)+Letters pattern (e.g., 1802 TW)
        match_space = re.match(r"^(\d+)\s+([A-Z]+)$", symbol) # \s+ matches one or more spaces
        if match_space:
            number_part = match_space.group(1)
            letter_part = match_space.group(2)
            normalized_symbol = f"{number_part}.{letter_part}"
            print(f"Detected Number+Space+Letters pattern, normalized to: {normalized_symbol}")
            return normalized_symbol

        # If none of the specific patterns matched, return cleaned input
        # Let yfinance try to interpret standard tickers (AAPL) or others.
        print(f"Symbol format not automatically normalized, using as entered: {symbol}")
        return symbol

    def select_date_range(self):
        """Presents a dialog for selecting a RELATIVE date range."""
        self.dates = None # Reset dates
        self.date_window = tk.Toplevel(self.root)
        self.date_window.title("Select Display Period") # Simplified title

        # --- Window Setup ---
        self.date_window.update_idletasks()
        width = 300 # Can be a bit smaller now
        height = 180
        x = (self.date_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.date_window.winfo_screenheight() // 2) - (height // 2)
        self.date_window.geometry(f'{width}x{height}+{x}+{y}')
        self.date_window.resizable(False, False)

        # --- REMOVED: self.mode variable ---

        # --- Main Frame ---
        main_frame = tk.Frame(self.date_window)
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)

        # --- Relative Frame (Now the only frame) ---
        self.relative_frame = tk.Frame(main_frame)
        tk.Label(self.relative_frame, text="Select Display Period").grid(row=0, column=0, columnspan=2, pady=(0,10))

        tk.Label(self.relative_frame, text="Select unit:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.unit_var = tk.StringVar(value="Months")
        self.unit_menu = ttk.Combobox(self.relative_frame, textvariable=self.unit_var,
                                      values=["Days", "Months", "Years"], state="readonly", width=12) # Slightly wider
        self.unit_menu.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        tk.Label(self.relative_frame, text="Enter number:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.count_entry = tk.Entry(self.relative_frame, width=12) # Slightly wider
        self.count_entry.insert(0, "3") # Default value
        self.count_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

        # Pack the relative frame directly
        self.relative_frame.pack(fill="both", expand=True)
        self.relative_frame.columnconfigure(1, weight=1) # Make entry/combobox expand


        # --- Button Frame (Only Submit) ---
        button_frame = tk.Frame(main_frame)
        button_frame.pack(side="bottom", fill="x", pady=(15, 0)) # Add padding above


        # Submit button centered or right-aligned
        self.submit_btn = ttk.Button(button_frame, text="Submit", command=self.submit_dates, style="Accent.TButton")
        self.submit_btn.pack(pady=5) # Center it by default, or use side="right"

        # Set initial focus
        self.count_entry.focus_set()
        self.count_entry.select_range(0, tk.END)

        # Handle closing via 'X' button
        self.date_window.protocol("WM_DELETE_WINDOW", self.cancel_date_selection)

        # Make window modal
        self.date_window.grab_set()
        self.root.wait_window(self.date_window)
        return self.dates

    # --- New helper for cancelling date selection ---
    def cancel_date_selection(self):
        """Handles closing the date window without submitting."""
        print("DEBUG: Date selection cancelled.")
        self.dates = None # Ensure dates is None if cancelled
        if self.date_window and self.date_window.winfo_exists():
            self.date_window.destroy()

    def submit_dates(self):
        """Processes the RELATIVE date selection."""
        self.dates = {}
        end_date = datetime.today().date()
        start_date = None

        # --- No mode check needed ---
        try:
            count = int(self.count_entry.get())
            if count <= 0:
                raise ValueError("Count must be positive.")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid positive integer for number.", parent=self.date_window)
            return # Keep window open for correction

        unit = self.unit_var.get()

        try: # Add try-except for date calculation just in case
            if unit == "Days":
                start_date = end_date - timedelta(days=count)
            elif unit == "Months":
                start_date = end_date - pd.DateOffset(months=count)
                start_date = start_date.date() # Convert back to date object
            elif unit == "Years":
                start_date = end_date - pd.DateOffset(years=count)
                start_date = start_date.date() # Convert back to date object
            else:
                 messagebox.showerror("Error", "Invalid unit selected.", parent=self.date_window)
                 return
        except Exception as e:
             messagebox.showerror("Error", f"Error calculating date: {e}", parent=self.date_window)
             return

        # Basic sanity check (optional, but good)
        if start_date >= end_date:
             messagebox.showerror("Error", f"Calculated start date ({start_date}) is not before end date ({end_date}). Please choose a smaller period.", parent=self.date_window)
             return

        # --- END of relative mode logic ---

        self.dates['start'] = start_date
        self.dates['end'] = end_date
        print(f"DEBUG: Dates submitted - Start: {start_date}, End: {end_date}") # Log submitted dates
        if self.date_window and self.date_window.winfo_exists():
            self.date_window.destroy() # Close window on successful submission

    def show_loading_window(self, message="Fetching news data...\nPlease wait."):
        """Creates and displays a non-closable, centered loading window."""
        loading_window = tk.Toplevel(self.root)
        loading_window.title("Loading")
        # Remove window decorations (title bar, borders)
        loading_window.overrideredirect(True)
        loading_window.geometry("300x120")  # Set a fixed size
        loading_window.configure(bg="white", relief=tk.SOLID, borderwidth=1)  # White bg, thin border

        # Keep window on top
        loading_window.transient(self.root)
        loading_window.attributes("-topmost", True)

        # Create a frame for content
        frame = tk.Frame(loading_window, bg="white")
        frame.pack(expand=True, fill="both", padx=20, pady=10)

        # Loading label
        loading_label = ttk.Label(frame, text=message, font=("Arial", 11),
                                  background="white", justify=tk.CENTER)
        loading_label.pack(pady=(10, 5))

        # Progress bar
        progress_bar = ttk.Progressbar(frame, mode='indeterminate', length=250)
        progress_bar.pack(pady=(5, 10))
        progress_bar.start(15)  # Start the animation (interval in ms)

        # Center the window on the screen
        loading_window.update_idletasks()  # Ensure window size is calculated
        screen_width = loading_window.winfo_screenwidth()
        screen_height = loading_window.winfo_screenheight()
        x = (screen_width // 2) - (loading_window.winfo_width() // 2)
        y = (screen_height // 2) - (loading_window.winfo_height() // 2)
        loading_window.geometry(f"+{x}+{y}")

        # Make it modal (prevents interaction with other windows)
        loading_window.grab_set()

        # Force UI update to show the window immediately
        loading_window.update()

        return loading_window  # Return the window object

    def compute_indicators(self, df, vol_window=14, ema_window=20):
        """Computes technical indicators needed for the LSTM model."""
        # Initial check for empty DataFrame
        if df.empty:
             print("Warning: Input DataFrame is empty for indicator calculation.")
             return pd.DataFrame(columns=["close", "EMA20", "MACD", "RSI_MA", "RSI_Slope", "OBV", "ATR", "StdDev", "BBWidth"])

        data = df.copy()
        # Robust lowercasing
        try:
            data.columns = [str(col).lower() for col in data.columns]
        except Exception as e:
            print(f"ERROR: Failed to lowercase columns: {e}")
            return pd.DataFrame(columns=["close", "EMA20", "MACD", "RSI_MA", "RSI_Slope", "OBV", "ATR", "StdDev", "BBWidth"])
        print(f"DEBUG compute_indicators: Columns after lowercasing: {data.columns.tolist()}")

        # ---> ADD THIS: Drop duplicate columns, keeping the first occurrence <---
        original_cols = data.columns.tolist()
        data = data.loc[:, ~data.columns.duplicated(keep='first')]
        new_cols = data.columns.tolist()
        if len(original_cols) != len(new_cols):
            print(f"WARNING: Duplicate columns dropped. Original: {original_cols}, Kept: {new_cols}")
        # ----------------------------------------------------------------------

        # --- Check for REQUIRED columns AFTER deduplication ---
        required_cols = ['close', 'high', 'low', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
             print(f"ERROR: Missing required columns AFTER deduplication: {missing_cols}. Cannot compute indicators.")
             return pd.DataFrame(columns=["close", "EMA20", "MACD", "RSI_MA", "RSI_Slope", "OBV", "ATR", "StdDev", "BBWidth"])
        # ----------------------------------------------------

        # --- Now calculations should use unique columns ---
        try: # Wrap calculations in try block for safety
            # EMA20
            data["EMA20"] = data["close"].ewm(span=ema_window, adjust=False).mean()

            # MACD
            ema12 = data["close"].ewm(span=12, adjust=False).mean()
            ema26 = data["close"].ewm(span=26, adjust=False).mean()
            data["MACD"] = ema12 - ema26

            # RSI
            rsi_period = 14
            data["RSI"] = compute_rsi(data["close"], period=rsi_period) # Assumes compute_rsi handles Series input
            rsi_ma_length = 9
            data["RSI_MA"] = data["RSI"].rolling(window=rsi_ma_length).mean()
            data["RSI_Slope"] = data["RSI_MA"].diff()

            # OBV (On Balance Volume)
            obv = [0] * len(data)
            # Use .loc to avoid potential SettingWithCopyWarning
            close_series = data['close']
            volume_series = data['volume']
            for i in range(1, len(data)):
                if close_series.iloc[i] > close_series.iloc[i-1]:
                    obv[i] = obv[i-1] + volume_series.iloc[i]
                elif close_series.iloc[i] < close_series.iloc[i-1]:
                    obv[i] = obv[i-1] - volume_series.iloc[i]
                else:
                    obv[i] = obv[i-1]
            data["OBV"] = obv
            # Scale OBV safely
            if not data[["OBV"]].empty:
                 data["OBV"] = MinMaxScaler().fit_transform(data[["OBV"]])
            else:
                 data["OBV"] = 0 # Or np.nan

            # ATR (Average True Range)
            high_low = data["high"] - data["low"]
            high_prev_close = (data["high"] - data["close"].shift()).abs()
            low_prev_close = (data["low"] - data["close"].shift()).abs()
            # Need to handle potential NaNs introduced by shift() before max()
            tr_df = pd.DataFrame({'hl': high_low, 'hpc': high_prev_close, 'lpc': low_prev_close})
            tr = tr_df.max(axis=1, skipna=True) # Use skipna=True
            data["ATR"] = tr.ewm(span=vol_window, adjust=False).mean()

            # Bollinger Bands related features
            data["StdDev"] = data["close"].rolling(window=vol_window).std()
            data["UpperBB"] = data["EMA20"] + (data["StdDev"] * 2)
            data["LowerBB"] = data["EMA20"] - (data["StdDev"] * 2)
            # Use .loc to avoid potential SettingWithCopyWarning when dividing
            bb_width_calc = (data["UpperBB"] - data["LowerBB"]) / data["EMA20"].replace(0, np.nan)
            # Use .bfill() correctly on the Series being assigned
            data.loc[:, "BBWidth"] = bb_width_calc.bfill() # Assign back using .loc

            # --- Feature Selection ---
            feature_cols = ["close", "EMA20", "MACD", "RSI_MA", "RSI_Slope", "OBV", "ATR", "StdDev", "BBWidth"]

            # Ensure selected columns actually exist before returning
            final_feature_cols = [col for col in feature_cols if col in data.columns]
            if len(final_feature_cols) != len(feature_cols):
                print(f"WARNING: Some feature columns missing after calculation: {set(feature_cols) - set(final_feature_cols)}")

            # Handle NaNs created by rolling calculations
            data_processed = data[final_feature_cols].dropna() # Operate on selected cols

            print(f"Computed Technical Indicators. Shape after dropna: {data_processed.shape}")
            if not data_processed.empty:
                print(data_processed.head())
            else:
                print("No data left after computing indicators and dropping NaNs.")
                return pd.DataFrame(columns=final_feature_cols)

            return data_processed # Return the processed DataFrame

        except KeyError as ke:
             print(f"ERROR: KeyError during indicator calculation, likely missing column: {ke}")
             traceback.print_exc()
             return pd.DataFrame(columns=["close", "EMA20", "MACD", "RSI_MA", "RSI_Slope", "OBV", "ATR", "StdDev", "BBWidth"])

        except Exception as e:
             print(f"ERROR during indicator calculation: {type(e).__name__} - {e}")
             traceback.print_exc()
             return pd.DataFrame(columns=["close", "EMA20", "MACD", "RSI_MA", "RSI_Slope", "OBV", "ATR", "StdDev", "BBWidth"]) # Return empty on error


    def get_company_name(self, symbol):
        try:
            print(f"DEBUG: Fetching Ticker info for {symbol}")
            stock = yf.Ticker(symbol)
            # Check if info dictionary is returned and not empty
            info = stock.info
            if info and isinstance(info, dict): # Check if info is a non-empty dict
                # Prioritize longName, then shortName, finally the symbol itself
                name = info.get("longName", info.get("shortName", symbol))
                final_name = name if name else symbol
                print(f"DEBUG: Found name: {final_name}")
                return final_name
            else:
                print(f"Warning: yf.Ticker({symbol}).info returned empty or invalid type ({type(info)}).")
                return symbol # Fallback to symbol if info is missing/invalid
        except Exception as e:
                # Catch potential errors during Ticker() creation or .info access
                print(f"Error fetching company name for {symbol}: {type(e).__name__} - {e}")
                # traceback.print_exc() # Optional: uncomment for more detail on network errors
                return symbol  # Fallback to the symbol on any error

    # Modify the function signature to accept the message variable
    def get_news_sentiment(self, symbol, loading_msg_var=None):
        """
        Fetch news sentiment data. Returns dict: date -> avg sentiment.
        Now includes zero scores in aggregation.
        Accepts a StringVar to update the loading message.
        """
        if not PERFORM_SENTIMENT:
            print("Skipping sentiment analysis due to missing libraries.")
            return {}

        # --- Update Loading Message (Optional) ---
        # Let the user know we are starting the GDELT fetch specifically
        if loading_msg_var:
            self.update_loading_message(loading_msg_var, f"Querying GDELT for {symbol} news...")

        # --- Keyword Strategy ---
        company_name = self.get_company_name(symbol)
        query_keyword = company_name
        print(f"DEBUG: Using Raw Company Name keyword for news query: '{query_keyword}'")

        # --- Date Range ---
        end_date = datetime.today().date()
        start_date = end_date - timedelta(days=60)

        # --- GDELT Fetch ---
        self.article_urls = []
        articles_df = pd.DataFrame()
        try:
            print("DEBUG [get_news_sentiment]: Starting GDELT search...")
            f_news = Filters(keyword=query_keyword, start_date=start_date.strftime("%Y-%m-%d"),
                             end_date=end_date.strftime("%Y-%m-%d"), num_records=200)
            gd_news = GdeltDoc()
            articles_df = gd_news.article_search(f_news)
            print(f"DEBUG: GDELT search returned {len(articles_df)} potential articles.")
        except Exception as e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"ERROR searching GDELT for '{query_keyword}': {type(e).__name__} - {e}")
            traceback.print_exc()
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            articles_df = pd.DataFrame()

        if articles_df.empty:
            print("DEBUG [get_news_sentiment]: No news articles found via GDELT.")
            # Update loading message if needed before returning
            if loading_msg_var:
                self.update_loading_message(loading_msg_var, f"No news found for {symbol} via GDELT.")
                # Optional: Add a small delay so user can see the message
                # time.sleep(1.5)
            return {}

        # --- Update Loading Message for Processing ---
        num_articles = len(articles_df)
        if loading_msg_var:
            self.update_loading_message(loading_msg_var, f"Processing {num_articles} potential articles...")

        # --- Sentiment Processing ---
        analyzer = SentimentIntensityAnalyzer()
        sentiments = {}
        processed_urls = set()
        processed_count = 0
        successful_count = 0 # Counter for successful articles

        print(f"DEBUG [get_news_sentiment]: Starting processing loop for {num_articles} articles...")
        for index, row in articles_df.iterrows():
            # --- Optional: Update loading message periodically ---
            processed_count += 1
            if loading_msg_var and processed_count % 10 == 0: # Update more frequently maybe
                self.update_loading_message(loading_msg_var, f"Processing article {processed_count}/{num_articles}...")

            url = row.get('url')
            if not url or not isinstance(url, str) or not url.startswith(('http://', 'https://')): # Basic URL validation
                 print(f"DEBUG [get_news_sentiment]: Skipping invalid or missing URL in row {index}: {url}")
                 continue
            if url in processed_urls:
                print(f"DEBUG [get_news_sentiment]: Skipping already processed URL: {url}")
                continue
            processed_urls.add(url)

            # --- START OF DETAILED DEBUGGING BLOCK FOR NEWSPAPER ---
            print(f"\nDEBUG [get_news_sentiment]: === Attempting URL ({processed_count}/{num_articles}) ===\nURL: {url}")

            text = None # Ensure text is reset for each article
            title = None # Ensure title is reset

            try:
                article = Article(url, config=newspaper_config)
                print(f"DEBUG [get_news_sentiment]: Downloading...")
                article.download()
                # Safely check status AFTER download attempt
                http_status_code = getattr(article, 'http_status', 'N/A') # Default to 'N/A' if not found
                print(f"DEBUG [get_news_sentiment]: Download state: {article.download_state}, HTTP Status: {http_status_code}")
                if article.download_state != 2:
                    print(f"DEBUG [get_news_sentiment]: ---> Download FAILED/SKIPPED (State: {article.download_state}, Status: {http_status_code})")
                    continue
                article.parse()
                text = article.text
                title = article.title
            except Exception as e:
                print(f"DEBUG [get_news_sentiment]: ---> EXCEPTION during download/parse for {url}:")
                traceback.print_exc() # This prints the full error details to console
                print("-" * 30) # Separator
                continue
            if not text or len(text) < 100:
                print(f"DEBUG [get_news_sentiment]: ---> SKIPPED: Text too short ({len(text) if text else 0} chars) or empty.")
                continue
            print(f"DEBUG [get_news_sentiment]: ---> SKIPPED: Text too short ({len(text) if text else 0} chars) or empty.")
            self.article_urls.append(url)

            try:
                score = analyzer.polarity_scores(text)['compound']
                print(f"DEBUG [get_news_sentiment]: ---> VADER Score: {score:.4f}")
            except Exception as e:
                print(f"DEBUG [get_news_sentiment]: ---> EXCEPTION during VADER scoring:")
                traceback.print_exc()
                print("-" * 30)
                continue

            pub_date_str = row.get('seendate') # GDELT often uses 'seendate'
            if pd.notnull(pub_date_str):
                try:
                    # Try parsing GDELT's typical format, handle potential errors
                    pub_date = pd.to_datetime(pub_date_str, format='%Y%m%dT%H%M%SZ', errors='coerce').date()
                    if pub_date is None: # Handle if format doesn't match exactly
                         print(f"DEBUG [get_news_sentiment]: ---> Could not parse date '{pub_date_str}' with expected format. Trying generic parse...")
                         pub_date = pd.to_datetime(pub_date_str, errors='coerce').date()

                    if pub_date is None: # If generic parse also failed
                         print(f"DEBUG [get_news_sentiment]: ---> Failed to parse date '{pub_date_str}' definitively.")
                         continue

                except (ValueError, TypeError, OverflowError) as e: # Catch more date errors
                    print(f"DEBUG [get_news_sentiment]: ---> EXCEPTION parsing date '{pub_date_str}': {type(e).__name__} - {e}")
                    continue
            else:
                print(f"DEBUG [get_news_sentiment]: ---> Missing 'seendate' in GDELT data for URL.")
                continue

            sentiments.setdefault(pub_date, []).append(score)
            successful_count += 1 # Increment success counter
            print(f"DEBUG [get_news_sentiment]: ---> Added score for date {pub_date}. Total successful: {successful_count}")
            # --- END OF PROCESSING FOR ONE ARTICLE ---
            print(f"\nDEBUG [get_news_sentiment]: Finished processing loop. Successfully processed {successful_count} articles.")

        # --- Update Loading Message - Aggregating ---
        if loading_msg_var:
            self.update_loading_message(loading_msg_var, "Aggregating sentiment scores...")

        # --- Calculate Daily Average Sentiment ---
        sentiment_avg = {}
        if sentiments:
            sentiment_avg = {d: sum(scores) / len(scores) for d, scores in sentiments.items()}
            print("DEBUG [get_news_sentiment]: Aggregated Sentiment Data (Date: Avg Score):")
            for d in sorted(sentiment_avg):
                print(f"  {d}: {sentiment_avg[d]:.4f}")
        else:
            print("DEBUG [get_news_sentiment]: No articles yielded scores to aggregate.")
            if loading_msg_var:  # Update final status if no scores
                self.update_loading_message(loading_msg_var, "No articles yielded scores.")
                print(f"DEBUG [get_news_sentiment]: Final count of collected article URLs: {len(self.article_urls)}")
                # time.sleep(1.5)

        print(f"Collected {len(self.article_urls)} relevant article URLs.")
        # --- Update Loading Message - Finishing ---
        if loading_msg_var:
            self.update_loading_message(loading_msg_var, "Sentiment analysis complete.")
            # time.sleep(0.5) # Short delay before window closes

        return sentiment_avg

    def display_news_urls(self):
        """Display a window with the list of fetched news URLs."""
        if not self.article_urls:
            messagebox.showwarning("No News", "No relevant news URLs were successfully processed.",
                                   parent=self.plot_window)
            return

        # Create a new window relative to the plot window if it exists
        parent_window = self.plot_window if self.plot_window and self.plot_window.winfo_exists() else self.root

        url_window = tk.Toplevel(parent_window)
        url_window.title("Processed News URLs")
        url_window.geometry("600x400")

        # --- URL Frame ---
        frame = tk.Frame(url_window)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, width=80, height=20)
        for i, url in enumerate(self.article_urls):
            listbox.insert(tk.END, f"{i + 1}. {url}")
        listbox.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar.config(command=listbox.yview)

        # --- Close Button ---
        close_button = tk.Button(url_window, text="Close", command=url_window.destroy)
        close_button.pack(pady=5)

        url_window.grab_set()  # Make modal

    def forecast_future_sentiment(self, future_dates):
        """
        Forecast future sentiment using linear regression on recent sentiment.
        """
        if not self.sentiment_data:
            print("No historical sentiment data to forecast from.")
            return {d: 0 for d in future_dates}  # Return neutral sentiment

        # Use only recent sentiment (e.g., last 30 days) for forecasting
        recent_cutoff = (datetime.today() - timedelta(days=30)).date()
        hist_dates_ord = []
        hist_sentiments = []
        for d, s in sorted(self.sentiment_data.items()):
            if d >= recent_cutoff:
                hist_dates_ord.append(d.toordinal())
                hist_sentiments.append(s)

        if len(hist_dates_ord) < 5:  # Need minimum points for regression
            print(f"Insufficient recent sentiment data ({len(hist_dates_ord)} points) for reliable forecast.")
            # Fallback: use the average of available recent data or overall average if none recent
            fallback_sentiment = np.mean(hist_sentiments) if hist_sentiments else np.mean(
                list(self.sentiment_data.values())) if self.sentiment_data else 0
            print(f"Using fallback sentiment: {fallback_sentiment:.4f}")
            return {d: fallback_sentiment for d in future_dates}

        hist_dates_ord = np.array(hist_dates_ord).reshape(-1, 1)
        hist_sentiments = np.array(hist_sentiments)

        try:
            model = LinearRegression()
            model.fit(hist_dates_ord, hist_sentiments)

            future_sentiment = {}
            future_dates_ord = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
            predicted_sentiments = model.predict(future_dates_ord)

            # Clip predictions to reasonable bounds (e.g., -1 to 1 VADER range)
            predicted_sentiments = np.clip(predicted_sentiments, -1, 1)

            for d, pred in zip(future_dates, predicted_sentiments):
                future_sentiment[d] = pred

            print("Forecasted Future Sentiment:")
            for d, s in future_sentiment.items():
                print(f"  {d}: {s:.4f}")
            return future_sentiment

        except Exception as e:
            print(f"Error during sentiment forecasting: {e}")
            fallback_sentiment = np.mean(hist_sentiments)  # Use average of data used for fit
            return {d: fallback_sentiment for d in future_dates}

    def predict_future(self, training_data, time_step=60):
        # **Fixed prediction period**
        prediction_days = 10

        # **Fixed window sizes for consistency** (can be adjusted)
        vol_window = 20  # Consistent with indicator calculation if needed
        ema_window = 20

        # --- 1. Feature Engineering ---
        # Use the already defined compute_indicators function
        features_df = self.compute_indicators(training_data, vol_window=vol_window, ema_window=ema_window)

        if features_df.empty:
            print("ERROR: Cannot predict, feature calculation resulted in empty data.")
            # Return empty arrays if features can't be calculated
            last_date = training_data.index[-1].date() if not training_data.empty else datetime.today().date()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days,
                                         freq='B').date  # Business days
            return future_dates, np.array([])

        # --- 2. Data Scaling ---
        # Scale features and target ('close' price) separately
        target_col = 'close'
        feature_cols = [col for col in features_df.columns if col != target_col]

        # Ensure 'close' is the first column for easier handling later if needed
        cols_ordered = [target_col] + feature_cols
        features_df = features_df[cols_ordered]

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(features_df)

        # Extract scalers for inverse transform later
        # We need the scaler for the 'close' column specifically
        close_scaler = MinMaxScaler()
        close_scaler.min_, close_scaler.scale_ = scaler.min_[0], scaler.scale_[0]

        # --- 3. Prepare Sequences for LSTM ---
        X_train, y_train = [], []
        for i in range(time_step, len(scaled_data)):
            X_train.append(scaled_data[i - time_step:i, :])  # Use all scaled features for input
            y_train.append(scaled_data[i, 0])  # Target is the scaled 'close' price (index 0)
        X_train, y_train = np.array(X_train), np.array(y_train)

        if X_train.shape[0] == 0:
            print(f"ERROR: Not enough data ({len(scaled_data)} points) to create sequences with time_step={time_step}.")
            last_date = training_data.index[-1].date() if not training_data.empty else datetime.today().date()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days, freq='B').date
            return future_dates, np.array([])

        # --- 4. Define and Train LSTM Model ---
        # Consider adjusting model complexity based on data size/needs
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),  # (time_step, num_features)
            LSTM(50, return_sequences=True),  # Reduced units slightly
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),  # Optional hidden dense layer
            Dense(1)  # Output layer predicts the next 'close' price
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model (consider more epochs for real use, but keep low for speed here)
        print(f"Training LSTM model on data shape: {X_train.shape}")
        # Use validation split to monitor overfitting
        model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1, verbose=0)
        print("LSTM model training complete.")

        # --- 5. Predict Future Values ---
        last_sequence = scaled_data[-time_step:]  # Get the last 'time_step' observations
        current_batch = last_sequence.reshape((1, time_step, X_train.shape[2]))  # Reshape for prediction
        predicted_scaled_prices = []

        for i in range(prediction_days):
            # Predict the next step (scaled price)
            next_pred_scaled = model.predict(current_batch, verbose=0)[0, 0]
            predicted_scaled_prices.append(next_pred_scaled)

            # *** IMPORTANT: Update the input sequence for the next prediction ***
            # Create a placeholder for the next step's features.
            # We only predicted 'close' (index 0). We need to estimate other features.
            # Simplest approach: assume other features remain constant or use the last known value.
            # More complex: build separate models to predict other features (difficult).
            # Let's use the last known values for other features.
            new_row = current_batch[0, -1, :].copy()  # Get last row of features
            new_row[0] = next_pred_scaled  # Update the 'close' feature with the prediction
            new_row_reshaped = new_row.reshape((1, 1, X_train.shape[2]))  # Reshape new row

            # Append the new predicted step and remove the oldest step
            current_batch = np.append(current_batch[:, 1:, :], new_row_reshaped, axis=1)

        # --- 6. Inverse Transform Predicted Prices ---
        predicted_prices = close_scaler.inverse_transform(np.array(predicted_scaled_prices).reshape(-1, 1)).flatten()

        # --- 7. Adjust Forecast (Optional but often helpful) ---
        # Shift prediction to start smoothly from the last actual price
        last_actual_price = training_data["close"].iloc[-1]
        price_shift = last_actual_price - predicted_prices[0]
        adjusted_prices = predicted_prices + price_shift

        # --- 8. Generate Future Dates ---
        last_date = training_data.index[-1].date()
        # Use pandas date_range for business days
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days, freq='B').date

        # --- 9. Incorporate Sentiment Forecast ---
        final_adjusted_prices = adjusted_prices.copy()  # Start with price prediction
        if PERFORM_SENTIMENT and self.sentiment_data:
            future_sentiment = self.forecast_future_sentiment(future_dates)
            if future_sentiment:
                # Apply sentiment adjustment (keep weight small initially)
                sentiment_weight = 0.005  # Small impact factor
                print("Applying sentiment adjustment to forecast...")
                for i, d in enumerate(future_dates):
                    sentiment_value = future_sentiment.get(d, 0)  # Use 0 if no forecast for a date
                    # Adjust price: increase if positive sentiment, decrease if negative
                    adjustment_factor = 1 + (sentiment_weight * sentiment_value)
                    final_adjusted_prices[i] *= adjustment_factor
            else:
                print("No sentiment forecast available to adjust prices.")
        else:
            print("Skipping sentiment adjustment.")

        # --- 10. Add Noise (Optional - for visual volatility) ---
        # Adding noise can make predictions look more realistic but reduces accuracy clarity
        # Keeping it moderate as per original code
        volatility_factor = final_adjusted_prices.mean() * 0.005  # Noise proportional to price level, smaller factor
        noise = np.random.normal(0, volatility_factor, size=final_adjusted_prices.shape)
        noisy_final_prices = final_adjusted_prices + noise

        # Ensure prices don't go below zero (or a reasonable floor)
        noisy_final_prices = np.maximum(noisy_final_prices, 0.01)

        print("Future Prediction Complete.")
        print("Predicted Prices (first 5):", noisy_final_prices[:5])

        return future_dates, noisy_final_prices.reshape(-1, 1)  # Return as column vector

    def display_plot(self, symbol, start_date, end_date, display_data, training_data, training_years):
        # Close previous plot window if exists
        if self.plot_window and self.plot_window.winfo_exists():
            # Ensure proper cleanup before destroying
            if hasattr(self, 'plot_figure') and self.plot_figure:
                plt.close(self.plot_figure)
                self.plot_figure = None
            if hasattr(self, 'plot_canvas_widget') and self.plot_canvas_widget:
                if self.plot_canvas_widget.winfo_exists():
                    self.plot_canvas_widget.destroy()
                self.plot_canvas_widget = None
            if hasattr(self, '_feedback_popup_job') and self._feedback_popup_job:
                try:
                    self.root.after_cancel(self._feedback_popup_job)
                except:
                    pass
                self._feedback_popup_job = None
            self.close_feedback_popup()

            # Update before destroying parent window
            self.plot_window.update_idletasks()
            self.plot_window.destroy()
            self.plot_window = None  # Reset reference

        self.plot_window = tk.Toplevel(self.root)
        self.plot_window.title(f"{symbol} Stock Price Analysis")

        # --- Set Initial Geometry, Resizability, and Minimum Size ---

        # 1. Get Screen Dimensions (requires brief window visibility)
        self.plot_window.geometry("1x1+0+0") # Place tiny window top-left temporarily
        self.plot_window.update_idletasks() # Process placement to read screen info
        screen_width = self.plot_window.winfo_screenwidth()
        screen_height = self.plot_window.winfo_screenheight()
        print(f"DEBUG: Detected Screen Resolution: {screen_width}x{screen_height}")

        # 2. Define Buffers/Margins for initial size calculation
        aspect_ratio = screen_width / screen_height if screen_height > 0 else 1.6 # Default to 16:10 if error
        print(f"DEBUG: Screen Aspect Ratio: {aspect_ratio:.3f}")

        # 3. Define Adjustment Parameters based on Aspect Ratio
        # Threshold: Treat ratios wider than 16:10 (1.6) as needing more vertical buffer.
        # Common wide ratio 16:9 is approx 1.778. Let's use 1.7 as a threshold.
        if aspect_ratio > 1.7: # Likely 16:9 or wider
            print("DEBUG: Applying adjustments for WIDER screen (e.g., 16:9)")
            plot_bottom_margin = 0.24 # Increase bottom margin significantly
            control_frame_pady = 3    # Reduce button frame padding more
            vertical_buffer = 110     # Increase initial vertical buffer (smaller initial window)
            title_fontsize = 11       # Slightly smaller title font
            title_pad = 20            # Slightly less title padding
            training_label_font_size = 9
            button_padx = 1           # Squeeze buttons horizontally slightly
            button_pady = (0, 1)      # Minimal vertical padding for button container
        else: # Likely 16:10, 4:3, etc.
            print("DEBUG: Applying adjustments for STANDARD/TALLER screen (e.g., 16:10)")
            plot_bottom_margin = 0.17 # Standard bottom margin
            control_frame_pady = 6    # Standard button frame padding
            vertical_buffer = 90      # Standard initial vertical buffer
            title_fontsize = 12       # Standard title font
            title_pad = 22           # Standard title padding
            training_label_font_size = 10
            button_padx = 2           # Standard horizontal button padding
            button_pady = (0, 2)      # Standard vertical padding for button container

        # 4. Calculate Target Initial Size using the chosen vertical_buffer
        horizontal_buffer = 30 # Keep horizontal buffer consistent
        initial_width = screen_width - horizontal_buffer
        initial_height = screen_height - vertical_buffer
        initial_width = max(800, initial_width) # Ensure reasonable minimums
        initial_height = max(550, initial_height) # Ensure reasonable minimums

        # 5. Set Initial Geometry
        x_pos = 5; y_pos = 5
        print(f"DEBUG: Setting initial geometry based on ratio: {initial_width}x{initial_height}+{x_pos}+{y_pos}")
        self.plot_window.geometry(f"{initial_width}x{initial_height}+{x_pos}+{y_pos}")

        # 6. Set Minimum Size (Consider adjusting hard_min_height based on ratio too if needed)
        min_width_ratio = 0.45; min_height_ratio = 0.40
        hard_min_width = 600; hard_min_height = 400 # Keep hard minimums for now
        calculated_min_width = int(screen_width * min_width_ratio)
        calculated_min_height = int(screen_height * min_height_ratio)
        final_min_width = max(hard_min_width, calculated_min_width)
        final_min_height = max(hard_min_height, calculated_min_height)
        print(f"DEBUG: Setting minimum size to: {final_min_width}x{final_min_height}")
        self.plot_window.minsize(final_min_width, final_min_height)

        # 7. Make Window Resizable
        self.plot_window.resizable(True, True)

        # 8. Final update before plotting
        self.plot_window.update_idletasks()

        # Add protocol handler
        self.plot_window.protocol("WM_DELETE_WINDOW", self.on_close)

        # --- Create Matplotlib Figure and Axes ---
        self.plot_figure, ax = plt.subplots(figsize=(12, 6))  # Keep figsize relatively standard
        self.plot_figure.subplots_adjust(
            left=0.08,
            right=0.85,  # Keep space for legend (adjust if needed)
            top=0.89,  # Slightly more space at top maybe
            # *** APPLY DYNAMIC BOTTOM MARGIN FROM RATIO CHECK ***
            bottom=plot_bottom_margin,
            hspace=0.3  # Add a bit of height space if you had multiple subplots later
        )

        # Plot historical data
        ax.plot(display_data.index, display_data["close"], label="Historical Close", color="cornflowerblue",
                linewidth=1.5)

        # --- Title and Labels ---
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        company_name = self.get_company_name(symbol)
        # *** APPLY DYNAMIC FONTSIZE AND PADDING ***
        ax.set_title(f"{company_name} ({symbol}) Stock Price\nDisplayed: {start_str} to {end_str}",
                     fontsize=title_fontsize, pad=title_pad)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel(f"Price ({'USD' if '.' not in symbol else symbol.split('.')[-1]})", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)

        # --- Predict Future ---
        # Show a brief message while predicting
        predict_label = tk.Label(self.plot_window, text="Generating prediction...", font=("Arial", 10),
                                 bg="lightyellow")
        predict_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.plot_window.update()  # Show the label immediately

        try:
            future_dates, predicted_prices = self.predict_future(training_data, time_step=60)
        finally:
            predict_label.destroy()  # Remove label

        # --- Plot Predicted Data ---
        if predicted_prices.size > 0:
            ax.plot(future_dates, predicted_prices, color="red", linestyle="--", marker='.', markersize=4,
                    label="Predicted Close")
        else:
            print("No prediction data to plot.")

        # --- Format Axes ---
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
        self.plot_figure.autofmt_xdate() # Use self.plot_figure here

        # --- Legend ---
        # Place legend outside plot area for clarity
        leg = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0., title="Legend")
        self.original_xlim = ax.get_xlim(); self.original_ylim = ax.get_ylim()

        # Store original limits for reset view
        self.original_xlim = ax.get_xlim()
        self.original_ylim = ax.get_ylim()

        # --- Prepare Data for Hover ---
        # Combine data *once* for faster lookup
        # Use numerical representation for dates for faster calculations
        hist_xdata_num = mdates.date2num(display_data.index.to_pydatetime())
        hist_ydata = display_data["close"].values.flatten()

        pred_xdata_num = np.array([])
        pred_ydata = np.array([])
        if predicted_prices.size > 0:
            # Convert python date objects to datetime objects first if needed
            future_dates_dt = [datetime.combine(d, datetime.min.time()) for d in future_dates]
            pred_xdata_num = mdates.date2num(future_dates_dt)
            pred_ydata = predicted_prices.flatten()

        # Combine for hover lookup
        combined_xdata_num = np.concatenate([hist_xdata_num, pred_xdata_num])
        combined_ydata = np.concatenate([hist_ydata, pred_ydata])
        # Sort combined data by date (numerical value) for binary search
        sort_indices = np.argsort(combined_xdata_num)
        combined_xdata_num_sorted = combined_xdata_num[sort_indices]
        combined_ydata_sorted = combined_ydata[sort_indices]

        # --- Tooltip Annotation ---
        self.annotation = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                                      textcoords="offset points",
                                      bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
                                      arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        self.annotation.set_visible(False)

        # --- OPTIMIZED Hover Event Handler ---
        def on_move(event):
            if event.inaxes != ax:
                if self.annotation.get_visible():
                    self.annotation.set_visible(False)
                    self.plot_figure.canvas.draw_idle()
                return

            if not combined_xdata_num_sorted.size: return  # No data to check

            # Use binary search (np.searchsorted) to find the index where event.xdata would be inserted
            # This is much faster (O(log N)) than iterating through all points (O(N))
            idx = np.searchsorted(combined_xdata_num_sorted, event.xdata)

            # Handle edge cases: cursor before the first point or after the last point
            if idx == 0:
                final_idx = 0
            elif idx == len(combined_xdata_num_sorted):
                final_idx = len(combined_xdata_num_sorted) - 1
            else:
                # Compare distance to the point before and the point at the insertion index
                dist_left = event.xdata - combined_xdata_num_sorted[idx - 1]
                dist_right = combined_xdata_num_sorted[idx] - event.xdata
                if dist_left < dist_right:
                    final_idx = idx - 1
                else:
                    final_idx = idx

            # Get the coordinates of the closest point
            x_closest_num = combined_xdata_num_sorted[final_idx]
            y_closest = combined_ydata_sorted[final_idx]

            # Update annotation position and text
            self.annotation.xy = (mdates.num2date(x_closest_num), y_closest)  # Use num2date for positioning
            date_str = mdates.num2date(x_closest_num).strftime("%Y-%m-%d")
            self.annotation.set_text(f"Date: {date_str}\nPrice: {y_closest:.2f}")

            # Make annotation visible and redraw if necessary
            if not self.annotation.get_visible():
                self.annotation.set_visible(True)
            self.plot_figure.canvas.draw_idle()

        # Connect the event handler
        self.plot_figure.canvas.mpl_connect("motion_notify_event", on_move)

        # --- Interactive Legend ---
        # Enable toggling lines by clicking legend items
        lines_map = {legline: origline for legline, origline in zip(leg.get_lines(), ax.get_lines())}

        def on_pick(event):
            legline = event.artist
            if legline in lines_map:
                origline = lines_map[legline]
                vis = not origline.get_visible()
                origline.set_visible(vis)
                # Dim the legend line based on visibility
                legline.set_alpha(1.0 if vis else 0.2)
                self.plot_figure.canvas.draw_idle()

        self.plot_figure.canvas.mpl_connect("pick_event", on_pick)
        for legline in leg.get_lines():
            legline.set_picker(True)  # Enable picking for legend lines
            legline.set_pickradius(5)  # Set tolerance for clicking

        # --- Zooming with Scroll Wheel ---
        def on_scroll(event):
            if event.inaxes != ax: return
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            xdata = event.xdata
            ydata = event.ydata
            if xdata is None or ydata is None: return  # Check if cursor is inside axes limits

            # Define zoom factor
            scale_factor = 1.3  # Zoom step
            if event.button == 'up':  # Zoom in
                new_width = (cur_xlim[1] - cur_xlim[0]) / scale_factor
                new_height = (cur_ylim[1] - cur_ylim[0]) / scale_factor
            elif event.button == 'down':  # Zoom out
                new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
                new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            else:  # Other scroll event?
                return

            # Calculate new limits centered on cursor position
            rel_x = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rel_y = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

            # Prevent zooming out beyond original data range
            orig_width = self.original_xlim[1] - self.original_xlim[0]
            orig_height = self.original_ylim[1] - self.original_ylim[0]

            # Clamp zoom out
            if new_width > orig_width: new_width = orig_width
            if new_height > orig_height: new_height = orig_height

            # Set new limits
            ax.set_xlim([xdata - new_width * (1 - rel_x), xdata + new_width * rel_x])
            ax.set_ylim([ydata - new_height * (1 - rel_y), ydata + new_height * rel_y])

            # Keep limits within original bounds when zooming out fully
            if new_width >= orig_width: ax.set_xlim(self.original_xlim)
            if new_height >= orig_height: ax.set_ylim(self.original_ylim)

            self.plot_figure.canvas.draw_idle()

        self.plot_figure.canvas.mpl_connect("scroll_event", on_scroll)

        # --- Embed Plot in Tkinter ---
        canvas = FigureCanvasTkAgg(self.plot_figure, master=self.plot_window)
        self.plot_canvas_widget = canvas.get_tk_widget()
        # Pack Canvas FIRST
        self.plot_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()

        # --- Bottom Control Frame ---
        # *** APPLY DYNAMIC PADDING FROM RATIO CHECK ***
        control_frame = tk.Frame(self.plot_window, pady=control_frame_pady)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X) # Keep packing order

        button_container = tk.Frame(control_frame)
        # *** Pack this first, anchor to RIGHT, NO expand/fill needed here ***
        button_container.pack(side=tk.RIGHT, padx=5, pady=button_pady)

        # Left-aligned label
        training_years_label = tk.Label(control_frame,
                                        text=f"Training Period: {training_years} years",
                                        font=("Arial", training_label_font_size))
        training_years_label.pack(side=tk.LEFT, padx=10, pady=(0, 3)) # Keep standard padding here

        # *** APPLY DYNAMIC PADX FOR BUTTONS ***
        reselect_button = ttk.Button(button_container, text="Reselect Symbol", command=self.reselect_symbol)
        reselect_button.pack(side=tk.LEFT, padx=button_padx) # Apply dynamic padx

        reset_button = ttk.Button(button_container, text="Reset View", command=lambda: self.reset_view(ax, self.plot_figure))
        reset_button.pack(side=tk.LEFT, padx=button_padx) # Apply dynamic padx

        doc_button = ttk.Button(button_container, text="Documentation", command=self.open_documentation)
        doc_button.pack(side=tk.LEFT, padx=button_padx) # Apply dynamic padx

        feedback_button = ttk.Button(button_container, text="Feedback", command=self.open_feedback_form)
        feedback_button.pack(side=tk.LEFT, padx=button_padx) # Apply dynamic padx

        save_button = ttk.Button(button_container, text="Save Plot", command=lambda: self.save_plot(self.plot_figure))
        save_button.pack(side=tk.LEFT, padx=button_padx) # Apply dynamic padx

        if PERFORM_SENTIMENT:
            news_button = ttk.Button(button_container, text="Show News URLs", command=self.display_news_urls)
            news_button.pack(side=tk.LEFT, padx=button_padx) # Apply dynamic padxnamic padx

        # --- Final window adjustments ---
        self.plot_window.lift()
        self.plot_window.focus_set()
        print("DEBUG: Plot window displayed and focused.")
        self._feedback_popup_job = self.plot_window.after(50, self.show_feedback_popup)

    def reset_view(self, ax, fig):
        """Resets the plot view to the original zoom level."""
        if hasattr(self, 'original_xlim') and self.original_xlim and \
           hasattr(self, 'original_ylim') and self.original_ylim:
            ax.set_xlim(self.original_xlim)
            ax.set_ylim(self.original_ylim)
            fig.canvas.draw_idle()
        else:
            print("DEBUG: Original limits not stored, cannot reset view.")

    def save_plot(self, fig):
        """Saves the current plot to an image file."""
        if not self.plot_window or not self.plot_window.winfo_exists():
            messagebox.showerror("Error", "Plot window is not available.", parent=self.root)
            return
        if not fig:  # Check if figure object is valid
            messagebox.showerror("Error", "Plot figure reference is missing.", parent=self.plot_window)
            return
        file_path = filedialog.asksaveasfilename(
            parent=self.plot_window,  # Set parent for modality
            title="Save plot image",
            defaultextension=".png",  # Default to PNG for better quality
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')  # Use the passed 'fig' object
                messagebox.showinfo("Saved", f"Plot saved successfully as:\n{file_path}", parent=self.plot_window)
            except Exception as e:
                messagebox.showerror("Error", f"Error saving the image: {e}", parent=self.plot_window)

    def reselect_symbol(self):
        """Closes the current plot and restarts the symbol selection process."""
        print("DEBUG: Reselect Symbol triggered.")

        # --- Cancel pending feedback popup job FIRST ---
        if hasattr(self, '_feedback_popup_job') and self._feedback_popup_job:
            try:
                self.root.after_cancel(self._feedback_popup_job)
            except Exception as e:
                print(f"DEBUG: Error/Warning cancelling feedback job: {e}")
            finally:
                 self._feedback_popup_job = None

        # --- Close existing feedback popup ---
        self.close_feedback_popup()

        # --- Destroy Plot Window and its contents ---
        if self.plot_window and self.plot_window.winfo_exists():
            print("DEBUG: Preparing to destroy plot window and contents.")
            try:
                # ---> 1. Explicitly Clean Up Matplotlib Figure <---
                #    Need a reference to the figure object ('fig') created in display_plot
                if hasattr(self, 'plot_figure') and self.plot_figure:
                    print("DEBUG: Closing Matplotlib figure...")
                    plt.close(self.plot_figure) # Ask Matplotlib to close the figure
                    print("DEBUG: Matplotlib figure closed.")
                    # Clear the reference immediately
                    self.plot_figure = None
                else:
                     print("DEBUG: No plot_figure reference found.")
                # ----------------------------------------------------

                # ---> 2. Destroy Canvas Widget (after closing figure) <---
                if hasattr(self, 'plot_canvas_widget') and self.plot_canvas_widget:
                     if self.plot_canvas_widget.winfo_exists():
                          print("DEBUG: Destroying plot canvas widget...")
                          # No need to update canvas if figure is closed first
                          self.plot_canvas_widget.destroy()
                          print("DEBUG: Plot canvas widget destroyed.")
                     else:
                          print("DEBUG: Plot canvas widget ref exists but widget already destroyed.")
                else:
                     print("DEBUG: No plot canvas widget reference found to destroy.")
                # --------------------------------------------------------

                # ---> 3. Force Tkinter Update Cycle <---
                #    Give Tkinter a chance to process the destruction events
                print("DEBUG: Updating root window after destroying canvas/figure...")
                self.root.update_idletasks()
                self.root.update()
                print("DEBUG: Root window updated.")
                # ---------------------------------------

                # 4. Now destroy the Toplevel window
                print("DEBUG: Destroying plot window.")
                self.plot_window.destroy()
                print("DEBUG: Plot window destroyed.")

            except Exception as e:
                 print(f"ERROR destroying plot elements in reselect_symbol: {e}")
                 traceback.print_exc()
                 # Attempt to destroy plot window even if other steps failed
                 try:
                      if self.plot_window and self.plot_window.winfo_exists():
                          self.plot_window.destroy()
                 except: pass # Ignore errors during final cleanup attempt

            finally:
                 # 5. Reset references
                 self.plot_window = None
                 self.plot_canvas_widget = None
                 self.plot_figure = None # Reset figure reference
                 print("DEBUG: plot_window references reset.")
        else:
             print("DEBUG: Plot window already destroyed or never created.")
             self.plot_window = None
             if hasattr(self, 'plot_canvas_widget'): self.plot_canvas_widget = None
             if hasattr(self, 'plot_figure'): self.plot_figure = None

        # --- Restart the analysis steps ---
        print("DEBUG: Scheduling run_analysis_steps after reselect.")
        self.root.after(50, self.run_analysis_steps)

    def on_close(self):
        """Handles closing of ANY main application window and shuts down."""
        print("DEBUG: on_close triggered. Preparing to shut down.")

        # --- 1. Cancel pending Tkinter .after jobs ---
        print("DEBUG: Cancelling pending feedback job (if any)...")
        if hasattr(self, '_feedback_popup_job') and self._feedback_popup_job:
            try:
                self.root.after_cancel(self._feedback_popup_job)
            except Exception as e:
                print(f"DEBUG: Error/Warning cancelling feedback job: {e}")
            finally:
                 self._feedback_popup_job = None
        # Add cancellations for any OTHER .after jobs you might have scheduled

        # --- 2. Destroy Modeless Top-Level Windows FIRST ---
        #    (Windows that don't block others, like the feedback popup)
        print("DEBUG: Destroying known modeless toplevels...")
        windows_to_close = [
            getattr(self, 'feedback_popup', None),
            # Add any other non-modal Toplevels here
        ]
        for window in windows_to_close:
            if window and isinstance(window, tk.Toplevel) and window.winfo_exists():
                try:
                    print(f"DEBUG: Destroying {type(window).__name__}...")
                    window.destroy()
                except Exception as e: print(f"DEBUG: Error destroying {type(window).__name__}: {e}")
        # Give Tkinter a moment to process these destructions
        if self.root and self.root.winfo_exists():
            try:
                 self.root.update()
            except: pass # Ignore error if root already gone


        # --- 3. Destroy the Main Application Content Window (Plot Window) ---
        #    (And its complex children like the canvas)
        print("DEBUG: Destroying plot window and its contents...")
        plot_window_ref = getattr(self, 'plot_window', None)
        canvas_widget_ref = getattr(self, 'plot_canvas_widget', None)
        figure_ref = getattr(self, 'plot_figure', None)
        # Clear instance refs immediately
        self.plot_window = None
        self.plot_canvas_widget = None
        self.plot_figure = None

        if plot_window_ref and plot_window_ref.winfo_exists():
             try:
                 if figure_ref:
                      print("DEBUG: Closing Matplotlib figure...")
                      plt.close(figure_ref)
                      print("DEBUG: Matplotlib figure closed.")
                 if canvas_widget_ref and canvas_widget_ref.winfo_exists():
                      print("DEBUG: Destroying plot canvas widget...")
                      canvas_widget_ref.destroy()
                      print("DEBUG: Plot canvas widget destroyed.")
                 # Update before destroying parent
                 print("DEBUG: Updating plot window before destroying...")
                 plot_window_ref.update()
                 print("DEBUG: Destroying plot window...")
                 plot_window_ref.destroy()
                 print("DEBUG: Plot window destroyed.")
             except Exception as e:
                  print(f"ERROR destroying plot elements in on_close: {e}")
                  traceback.print_exc()
        else:
             print("DEBUG: plot_window reference was None or window didn't exist.")


        # --- 4. Destroy Any Other Remaining Toplevels ---
        #    (e.g., date_window if it could possibly linger, though unlikely)
        other_toplevels = [getattr(self, 'date_window', None)]
        for window in other_toplevels:
             if window and isinstance(window, tk.Toplevel) and window.winfo_exists():
                  try: window.destroy()
                  except Exception as e: print(f"DEBUG: Error destroying other toplevel {type(window).__name__}: {e}")

        # --- 5. Destroy the Root Window LAST ---
        #    This terminates the mainloop cleanly
        try:
            if self.root and self.root.winfo_exists():
                print("DEBUG: Destroying root window (will stop mainloop).")
                self.root.destroy()
            else:
                 print("DEBUG: Root window already destroyed or never existed.")
        except Exception as e:
            print(f"DEBUG: Error destroying root window: {e}")

        # --- REMOVE sys.exit(0) ---
        # print("Application shutdown sequence complete. Forcing exit.")
        # sys.exit(0) # REMOVED
        # -------------------------
        print("Application shutdown sequence complete via on_close.")

    # Modify signature to accept the message variable
    def fetch_sentiment_data_threaded(self, symbol, loading_msg_var, callback):
        """Fetches sentiment data in a thread, updates loading message, and calls callback."""
        print("Starting sentiment data fetch thread...")
        self.sentiment_data = {}  # Reset sentiment data
        try:
            # ---> Pass loading_msg_var to get_news_sentiment <---
            self.sentiment_data = self.get_news_sentiment(symbol, loading_msg_var)
            print("Sentiment data fetch thread finished.")
        except Exception as e:
            print(f"Error in sentiment fetch thread: {e}")
            # Optionally update loading message with error
            if loading_msg_var:
                self.update_loading_message(loading_msg_var, f"Error during analysis:\n{e}")
                time.sleep(2)  # Show error briefly
        finally:
            # Schedule the callback to run in the main thread
            # Ensure root still exists before scheduling callback
            try:
                if self.root.winfo_exists():
                    self.root.after(0, callback)
            except tk.TclError:
                print("DEBUG: Root window destroyed before scheduling callback.")

    def update_loading_message(self, message_var, new_message):
        """ Safely updates the text of the loading window label. """
        if message_var:
            try:
                # Check if the root window (and thus the app) still exists
                if self.root.winfo_exists():
                    # Schedule the update in the main Tkinter thread
                    self.root.after(0, lambda: message_var.set(new_message))
            except Exception as e:
                print(f"DEBUG: Error updating loading message (window might be closing): {e}")

    def show_loading_window(self, initial_message="Initializing..."):
        """Creates and displays a non-closable, centered loading window."""
        loading_window = tk.Toplevel(self.root)
        loading_window.title("Processing")  # Title is hidden but good practice

        # --- Style and Behavior ---
        loading_window.overrideredirect(True)  # Borderless
        loading_window.resizable(False, False)
        loading_window.configure(bg="SystemButtonFace", relief=tk.SOLID, borderwidth=1)  # OS-native background
        loading_window.attributes("-topmost", True)  # Keep on top of other *app* windows

        # --- Content Frame ---
        frame = tk.Frame(loading_window, bg="SystemButtonFace", padx=20, pady=15)
        frame.pack(expand=True, fill="both")

        # --- Message Label (Store for Updates) ---
        # Use a StringVar for easier text updates
        self.loading_message_var = tk.StringVar(value=initial_message)
        loading_label = ttk.Label(frame, textvariable=self.loading_message_var,
                                  font=("Segoe UI", 10),  # Modern font
                                  background="SystemButtonFace", justify=tk.CENTER,
                                  wraplength=280)  # Wrap long messages
        loading_label.pack(pady=(0, 10), fill=tk.X)

        # --- Progress Bar ---
        progress_bar = ttk.Progressbar(frame, mode='indeterminate', length=280)
        progress_bar.pack(pady=(0, 5))
        progress_bar.start(20)  # Slower animation might feel less frantic

        # --- Centering ---
        loading_window.update_idletasks()  # Calculate initial size needed
        width = loading_window.winfo_reqwidth()  # Get calculated width
        height = loading_window.winfo_reqheight()  # Get calculated height
        # print(f"DEBUG: Loading window requested size: {width}x{height}") # Debug size

        screen_width = loading_window.winfo_screenwidth()
        screen_height = loading_window.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        loading_window.geometry(f'{width}x{height}+{x}+{y}')

        # --- Modality & Display ---
        loading_window.grab_set()  # Make modal relative to the application
        loading_window.lift()  # Ensure it's raised initially
        loading_window.update()  # Force display

        # Return window AND the message variable for updates
        return loading_window, self.loading_message_var

    def run_analysis_steps(self):
        """Guides the user through the steps, handling invalid symbols and restarting if data fetch fails."""

        # ---> Wrap major steps in a loop to allow restarting <---
        while True:
            print("\n--- Starting Input Sequence ---")  # Mark start of loop attempt

            # --- Step 1: Get Symbol (Using simpledialog) ---
            symbol_input = simpledialog.askstring(
                "Input",
                "Enter the stock symbol (e.g., AAPL, MSFT, 1155.KL, 0700.HK):",
                parent=self.root
            )
            if symbol_input is None:  # User cancelled
                print("DEBUG: Symbol input cancelled. Exiting.")
                self.on_close();
                return

            symbol = symbol_input.strip().upper()
            if not symbol:  # If empty after stripping
                messagebox.showwarning("Input Required", "Stock symbol cannot be empty. Please try again.",
                                       parent=self.root)
                continue  # Restart loop

            # Apply normalization logic directly
            if '.' not in symbol:
                # ... (Normalization logic using re.match for Number+Letter and Number+Space+Letter) ...
                match_no_space = re.match(r"^(\d+)([A-Z]+)$", symbol)
                if match_no_space:
                    symbol = f"{match_no_space.group(1)}.{match_no_space.group(2)}"
                    print(f"Detected Number+Letters pattern, normalized to: {symbol}")
                else:
                    match_space = re.match(r"^(\d+)\s+([A-Z]+)$", symbol)
                    if match_space:
                        symbol = f"{match_space.group(1)}.{match_space.group(2)}"
                        print(f"Detected Number+Space+Letters pattern, normalized to: {symbol}")
                    else:
                        print(f"Symbol format not automatically normalized, using as entered: {symbol}")
            else:
                print(f"Symbol contains '.', using as entered: {symbol}")
            # 'symbol' holds potentially normalized symbol

            # --- Step 2: Get Dates ---
            dates = self.select_date_range()
            if dates is None:
                print("DEBUG: Date selection cancelled. Restarting input sequence.")
                messagebox.showinfo("Restarting", "Date selection cancelled. Please start over.", parent=self.root)
                continue

            start_date = dates['start'];
            end_date = dates['end']
            start_str = start_date.strftime("%Y-%m-%d");
            end_str = end_date.strftime("%Y-%m-%d")

            # --- Step 3: Get Display Data (Check for Empty) ---
            display_data = pd.DataFrame()  # Initialize empty
            fetch_display_success = False
            loading_disp = None
            try:
                loading_disp, _ = self.show_loading_window(f"Fetching display data for {symbol}...")
                self.root.update()
                display_data = get_stock_data(symbol, start_str, end_str)  # Call the (potentially refined) function
                if not display_data.empty:  # Check return value
                    fetch_display_success = True
            except Exception as e:
                print(f"ERROR during display data fetch call: {e}")  # Error calling get_stock_data
                traceback.print_exc()
                messagebox.showerror("Fetch Error", f"An unexpected error occurred before fetching display data:\n{e}",
                                     parent=self.root)
                fetch_display_success = False
            finally:
                if loading_disp and loading_disp.winfo_exists():
                    loading_disp.grab_release();
                    loading_disp.destroy()

            # ---> Check if Display data fetch failed <---
            if not fetch_display_success:
                print(f"DEBUG: Display data fetch failed or empty for '{symbol}'. Restarting input sequence.")
                messagebox.showerror("Invalid Symbol or Data Error",  # More specific title
                                     f"Could not retrieve display data for symbol '{symbol}'.\n\n"
                                     f"Please verify the stock symbol is valid on Yahoo Finance and data exists for the selected period.",
                                     parent=self.root)
                continue  # ---> Restart the while loop (go back to Step 1)
            # --------------------------------------------

            print("DEBUG: Display data fetched successfully.")

            # --- Step 4: Get Training Period ---
            training_years = simpledialog.askinteger(
                "Training Data",
                "Enter number of years for LSTM training data:",
                parent=self.root, minvalue=1, initialvalue=3
            )
            if training_years is None:
                print("DEBUG: Training years cancelled. Restarting input sequence.")
                messagebox.showinfo("Restarting", "Training period selection cancelled. Please start over.",
                                    parent=self.root)
                continue

            # --- Step 5: Get Training Data (Check for Empty) ---
            today = datetime.today().date()
            training_start_date_dt = pd.Timestamp(today) - pd.DateOffset(years=training_years)
            training_start_date_str = training_start_date_dt.strftime("%Y-%m-%d")
            today_str = today.strftime("%Y-%m-%d")

            training_data = pd.DataFrame()  # Initialize empty
            fetch_training_success = False
            loading_train = None
            try:
                loading_train, _ = self.show_loading_window(f"Fetching {training_years} years training data...")
                self.root.update()
                training_data = get_stock_data(symbol, training_start_date_str, today_str)  # Call again
                if not training_data.empty:  # Check return value
                    fetch_training_success = True
            except Exception as e:
                print(f"ERROR during training data fetch call: {e}")  # Error calling get_stock_data
                traceback.print_exc()
                messagebox.showerror("Fetch Error", f"An unexpected error occurred before fetching training data:\n{e}",
                                     parent=self.root)
                fetch_training_success = False
            finally:
                if loading_train and loading_train.winfo_exists():
                    loading_train.grab_release();
                    loading_train.destroy()

            # ---> Check if Training data fetch failed <---
            if not fetch_training_success:
                print(f"DEBUG: Training data fetch failed or empty for '{symbol}'. Restarting input sequence.")
                messagebox.showerror("Data Error",
                                     f"Could not retrieve {training_years} years of training data for symbol '{symbol}'.\n\n"
                                     f"The symbol might be invalid, too new, or try a shorter training period.",
                                     parent=self.root)
                continue  # ---> Restart the while loop (go back to Step 1)
            # --------------------------------------------

            # --- If ALL data fetches successful, break the loop ---
            print("DEBUG: All required data fetched successfully. Breaking input loop.")
            break  # ---> Exit the 'while True' loop <---

        # --- Step 6: Fetch Sentiment (Threaded) & Display Plot ---
        # ALL variables needed below (symbol, start_date, end_date, etc.)
        # have been defined in the outer scope of this function by this point.
        loading_news_window = None
        loading_news_msg_var = None

        try:
            # Debug print to verify variables just before the 'if'
            print(
                f"DEBUG: Vars before Step 6 try: symbol={symbol}, start_date={start_date}, training_years={training_years}")

            if PERFORM_SENTIMENT:
                loading_news_window, loading_news_msg_var = self.show_loading_window(
                    f"Fetching news data for {symbol}...\nPlease wait."  # Uses symbol
                )

                # Define callback (accesses variables from outer scope)
                def after_sentiment_fetch():
                    if loading_news_window and loading_news_window.winfo_exists():
                        loading_news_window.grab_release()
                        loading_news_window.destroy()

                    print("Sentiment fetch complete. Proceeding to display plot.")
                    if not self.root.winfo_exists(): return
                    # Uses symbol, start_date, end_date, display_data, training_data, training_years
                    self.display_plot(symbol, start_date, end_date, display_data, training_data, training_years)

                # Start thread (uses symbol, loading_news_msg_var, after_sentiment_fetch)
                threading.Thread(
                    target=self.fetch_sentiment_data_threaded,
                    args=(symbol, loading_news_msg_var, after_sentiment_fetch),
                    daemon=True
                ).start()
            else:
                # ---> This block SHOULD have access to the variables <---
                print("Skipping sentiment analysis. Proceeding to display plot.")
                # Uses symbol, start_date, end_date, display_data, training_data, training_years
                self.display_plot(symbol, start_date, end_date, display_data, training_data, training_years)

        except Exception as e:
            print(f"Error occurred during sentiment fetch/plotting setup: {e}")
            if loading_news_window and loading_news_window.winfo_exists():
                loading_news_window.grab_release()
                loading_news_window.destroy()
            messagebox.showerror("Error", f"An error occurred during analysis setup:\n{e}", parent=self.root)

    # --- REVISED run() Method (Show Root Briefly) ---
    def run(self):
        """Shows initial dialog, withdraws root if OK, then starts app flow."""
        print("DEBUG: Entered run() method.")

        # --- Call the Introduction/Initial Dialog FIRST (while root is visible) ---
        choice = self.show_initial_dialog() # This now blocks until dialog closes

        print(f"DEBUG: Initial dialog choice returned: '{choice}'")

        # --- Check the User's Choice ---
        if choice != "ok":
            print(f"User cancelled or did not proceed (choice: '{choice}'). Exiting.")
            self.on_close() # Cleanup and quit
            return

        # --- Proceed if User Clicked OK ---
        print("User clicked OK.")

        # ---> Withdraw the root window NOW before scheduling main steps <---
        print("DEBUG: Withdrawing root window...")
        self.root.withdraw()
        self.root.update() # Process the withdraw event
        print("DEBUG: Root window withdrawn.")

        print("DEBUG: Proceeding to schedule analysis steps.")
        self.root.after(100, self.run_analysis_steps) # Schedule the main work

        # Start the Tkinter event loop
        print("DEBUG: Starting mainloop...")
        self.root.mainloop()
        print("DEBUG: mainloop finished.")

if __name__ == "__main__":
    try:
        app = StockAnalysisApp()
        app.run()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Optional: Log the full traceback

        traceback.print_exc()
    finally:
        print("Application finished.")