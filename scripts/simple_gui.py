#!/usr/bin/env python3
"""
Simple UAP Analyzer GUI
Basic tkinter interface for UAP video analysis
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import subprocess
import os
import sys
from pathlib import Path

class SimpleUAPGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🛸 UAP Video Analyzer")
        self.root.geometry("800x600")
        
        # Variables
        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="results/gui_analysis")
        self.analysis_running = False
        self.current_process = None
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="🛸 UAP Video Analyzer",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Video selection
        ttk.Label(main_frame, text="Video File:").grid(row=1, column=0, sticky=tk.W)
        
        video_entry = ttk.Entry(main_frame, textvariable=self.video_path, width=50)
        video_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 5))
        
        browse_btn = ttk.Button(main_frame, text="Browse", command=self.browse_video)
        browse_btn.grid(row=1, column=2, padx=(5, 0))
        
        # Output directory
        ttk.Label(main_frame, text="Output Dir:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        
        output_entry = ttk.Entry(main_frame, textvariable=self.output_dir, width=50)
        output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=(10, 0))
        
        browse_output_btn = ttk.Button(main_frame, text="Browse", command=self.browse_output)
        browse_output_btn.grid(row=2, column=2, padx=(5, 0), pady=(10, 0))
        
        # Analysis options
        options_frame = ttk.LabelFrame(main_frame, text="Analysis Options", padding="10")
        options_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(20, 0))
        options_frame.columnconfigure(0, weight=1)
        
        # Quick mode
        self.quick_mode = tk.BooleanVar(value=True)
        quick_cb = ttk.Checkbutton(options_frame, 
                                  text="Quick Mode (faster, core analysis only)",
                                  variable=self.quick_mode)
        quick_cb.grid(row=0, column=0, sticky=tk.W)
        
        # Advanced options
        self.atmospheric = tk.BooleanVar(value=True)
        self.physics = tk.BooleanVar(value=True)
        self.motion = tk.BooleanVar(value=True)
        
        adv_frame = ttk.Frame(options_frame)
        adv_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Checkbutton(adv_frame, text="Atmospheric Analysis", 
                       variable=self.atmospheric).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(adv_frame, text="Physics Analysis", 
                       variable=self.physics).grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        ttk.Checkbutton(adv_frame, text="Motion Analysis", 
                       variable=self.motion).grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=4, column=0, columnspan=3, pady=(20, 0))
        
        self.start_btn = ttk.Button(control_frame, text="🚀 Start Analysis", 
                                   command=self.start_analysis)
        self.start_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_btn = ttk.Button(control_frame, text="⏹️ Stop", 
                                  command=self.stop_analysis, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1)
        
        # Progress
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(20, 0))
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        # Status
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Log
        self.log_text = scrolledtext.ScrolledText(progress_frame, height=15, width=70)
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
    def browse_video(self):
        """Browse for video file"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.video_path.set(filename)
            
    def browse_output(self):
        """Browse for output directory"""
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_dir.set(dirname)
            
    def start_analysis(self):
        """Start video analysis"""
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file")
            return
            
        if not os.path.exists(self.video_path.get()):
            messagebox.showerror("Error", "Video file does not exist")
            return
            
        # Update UI
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.analysis_running = True
        self.status_label.config(text="Starting analysis...")
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        self.log("🚀 Starting UAP video analysis...")
        
        # Start analysis thread
        threading.Thread(target=self.run_analysis, daemon=True).start()
        
    def run_analysis(self):
        """Run the analysis in a separate thread"""
        try:
            # Build command
            cmd = [sys.executable, "run_advanced_analysis.py", self.video_path.get()]
            
            # Add output directory
            if self.output_dir.get():
                cmd.extend(["-o", self.output_dir.get()])
                
            # Add quick mode if selected
            if self.quick_mode.get():
                cmd.append("--quick")
                
            # Add specific analyses if not in quick mode
            if not self.quick_mode.get():
                if self.atmospheric.get():
                    cmd.append("--atmospheric")
                if self.physics.get():
                    cmd.append("--physics")
                    
            self.log(f"Command: {' '.join(cmd)}")
            
            # Run process
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True
            )
            
            # Read output
            while True:
                if not self.analysis_running:
                    break
                    
                output = self.current_process.stdout.readline()
                if output == '' and self.current_process.poll() is not None:
                    break
                    
                if output:
                    self.root.after(0, self.log, output.strip())
                    
                    # Update status based on output
                    if "Extracting frames" in output:
                        self.root.after(0, self.update_status, "Extracting frames...")
                    elif "motion" in output.lower():
                        self.root.after(0, self.update_status, "Analyzing motion...")
                    elif "physics" in output.lower():
                        self.root.after(0, self.update_status, "Physics analysis...")
                        
            # Check result
            return_code = self.current_process.wait()
            
            if return_code == 0:
                self.root.after(0, self.analysis_complete)
            else:
                self.root.after(0, self.analysis_error, f"Process failed with code {return_code}")
                
        except Exception as e:
            self.root.after(0, self.analysis_error, str(e))
            
    def stop_analysis(self):
        """Stop the analysis"""
        if self.current_process:
            self.current_process.terminate()
            
        self.analysis_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_status("Stopped")
        self.log("⏹️ Analysis stopped by user")
        
    def analysis_complete(self):
        """Handle successful completion"""
        self.analysis_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_status("✅ Analysis completed!")
        self.log("✅ Analysis completed successfully!")
        
        # Show completion message
        result = messagebox.askyesno("Analysis Complete", 
                                   "Analysis completed successfully!\n\nOpen results folder?")
        if result:
            self.open_results()
            
    def analysis_error(self, error_msg):
        """Handle analysis error"""
        self.analysis_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_status(f"❌ Error: {error_msg}")
        self.log(f"❌ Error: {error_msg}")
        
    def log(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        
    def update_status(self, status):
        """Update status label"""
        self.status_label.config(text=status)
        
    def open_results(self):
        """Open results folder"""
        output_dir = self.output_dir.get()
        if os.path.exists(output_dir):
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", output_dir])
            elif sys.platform == "win32":  # Windows
                subprocess.run(["explorer", output_dir])
            else:  # Linux
                subprocess.run(["xdg-open", output_dir])

def main():
    """Main function"""
    root = tk.Tk()
    app = SimpleUAPGUI(root)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()