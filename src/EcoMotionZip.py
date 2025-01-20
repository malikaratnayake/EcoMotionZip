import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import subprocess
import os
import threading
import tkinter.font as tkfont




class ConfigEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EcoMotionZip Video Processing Configuration")
        self.config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
        self.config_data = {}
        self.entries = {}

        # Create GUI layout
        self.create_widgets()

        # Load configuration at startup
        self.load_config()

    def create_widgets(self):
        # Configure root window to expand the canvas and frame
        self.root.columnconfigure(0, weight=1)  # Allow the canvas to expand horizontally
        self.root.rowconfigure(0, weight=1)     # Allow the canvas to expand vertically

        # Scrollable Frame
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Set larger dimensions for the canvas
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)  # Expand to fill available space
        self.scrollbar.grid(row=0, column=1, sticky="ns")  # Scrollbar next to canvas

        # Button frame
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # Center-align the buttons in the frame
        for i in range(4):  # Ensure there are 4 equal columns for the buttons
            self.button_frame.columnconfigure(i, weight=1)

        # Buttons
        ttk.Button(
            self.button_frame, text="Custom Configuration", command=self.load_custom_config
        ).grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        ttk.Button(
            self.button_frame, text="Save Configuration", command=self.save_config
        ).grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        ttk.Button(
            self.button_frame, text="Run EcoMotionZip", command=self.run_compression
        ).grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        ttk.Button(
            self.button_frame, text="Help", command=self.show_help
        ).grid(row=0, column=3, padx=5, pady=5, sticky="nsew")


    def append_output(self, text):
        """Append text to the output window."""
        self.output_text.configure(state="normal")
        self.output_text.insert("end", text + "\n")
        self.output_text.configure(state="disabled")
        self.output_text.see("end")

    def load_custom_config(self):
        """Allow the user to browse and load a custom JSON configuration file."""
        try:
            # Ensure updates to GUI before opening dialog
            self.root.update()
            file_path = filedialog.askopenfilename(
                title="Select Custom Configuration File",
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
            )
            if file_path:
                try:
                    with open(file_path, "r") as file:
                        self.config_data = json.load(file)  # Load JSON
                    self.config_file = file_path
                    self.root.after(100, self.populate_fields)  # Defer GUI updates
                    messagebox.showinfo("Success", f"Configuration loaded from {file_path}.")
                except json.JSONDecodeError:
                    messagebox.showerror("Error", "The selected file is not a valid JSON file.")
                except Exception as e:
                    messagebox.showerror("Error", f"An error occurred: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")


    def load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_file, "r") as f:
                self.config_data = json.load(f)

            self.populate_fields()
            # messagebox.showinfo("Success", f"Loaded configuration from {self.config_file}.")
        except FileNotFoundError:
            messagebox.showerror("Error", f"Configuration file {self.config_file} not found.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while loading the config: {e}")

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("Parameter Descriptions")
        help_window.geometry("400x400")

        # Add a scrollable text widget for help content
        text_widget = tk.Text(help_window, wrap="word", font=("Arial", 10))
        text_widget.pack(expand=True, fill="both")

        help_text = {
            "Video Source Directory": "Path to the directory containing the input videos.",
            "Output Directory": "Path to save the processed video files.",
            "Record Duration": "The duration in seconds for recording each video segment.",
            "Number of Videos": "The number of video segments to process.",
            "Camera Resolution": "Resolution of the camera, e.g., 1920x1080.",
            "Camera FPS": "Frames per second for the camera recording.",
            "Raspberry Pi Camera": "Toggle to enable or disable Raspberry Pi camera.",
            "Background Transparency": "Set the transparency level for the background.",
        }

        # Populate help text
        for param, description in help_text.items():
            text_widget.insert("end", f"{param}:\n{description}\n\n")

        text_widget.configure(state="disabled")  # Make the text read-only

    


    def populate_fields(self):
        """Populate GUI with configuration fields organized into sections with styled headings."""
        # Mapping JSON keys to descriptive labels
        descriptive_labels = {
            "video_source": "Video Source Directory",
            "output_directory": "Output Directory",
            "record_duration": "Record Duration (seconds)",
            "number_of_videos": "Number of Videos",
            "camera_resolution": "Camera Resolution (WxH)",
            "camera_fps": "Camera FPS",
            "raspberrypi_camera": "Use Raspberry Pi Camera",
            "delete_original_after_processing": "Delete Original After Processing",
            "embed_timestamps": "Embed Timestamps",
            "reader_sleep_seconds": "Reader Sleep Duration (seconds)",
            "reader_flush_proportion": "Reader Flush Proportion",
            "downscale_factor": "Downscale Factor",
            "dilate_kernel_size": "Dilation Kernel Size",
            "movement_threshold": "Movement Threshold",
            "post_motion_record_frames": "Post-Motion Record Frames",
            "full_frame_capture_interval": "Full Frame Capture Interval",
            "video_codec": "Video Codec",
            "num_opencv_threads": "Number of OpenCV Threads",
            "background_transparency": "Background Transparency",
            "save_frames": "Save Individual Frames",
            "frames_to_save": "Number of Frames to Save",
        }

        # Define sections and their corresponding keys
        sections = {
            "Directories": ["video_source", "output_directory"],
            "Output Configuration": ["delete_original_after_processing", "embed_timestamps", "video_codec", "background_transparency",
                                        "save_frames", "frames_to_save"],
            "Camera Controls": ["raspberrypi_camera", "camera_resolution", "camera_fps", "record_duration", "number_of_videos"],
            "Advanced Controls": [
            "downscale_factor", "dilate_kernel_size", "movement_threshold",
            "post_motion_record_frames", "full_frame_capture_interval",
            "reader_sleep_seconds", "reader_flush_proportion", "num_opencv_threads"
            ]
        }

        # Create a custom style for section headers
        header_font = tkfont.Font(family="Arial", size=15, weight="bold")
        style = ttk.Style()
        style.configure("Custom.TLabelframe.Label", font=header_font)

        # Clear existing widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self.entries.clear()
        row = 0

        # Loop through each section
        for section_name, keys in sections.items():
            # Hide the section with camera controls if the Raspberry Pi camera is not used
            if section_name == "Camera Controls":
                continue
            # Create a labeled frame for each section with custom style
            section_frame = ttk.LabelFrame(self.scrollable_frame, text=section_name, style="Custom.TLabelframe")
            section_frame.grid(row=row, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

            if section_name == "Advanced Controls":
                section_row = 0  # Row tracker
                col = 0  # Column tracker
                for i, key in enumerate(keys):
                    value = self.config_data.get(key, "")
                    label_text = descriptive_labels.get(key, key)

                    # Create label and widget
                    ttk.Label(section_frame, text=label_text).grid(row=section_row, column=col * 2, sticky="w", padx=5, pady=5)

                    if isinstance(value, bool):  # Checkbox for booleans
                        var = tk.BooleanVar(value=value)
                        chk = ttk.Checkbutton(section_frame, variable=var)
                        chk.grid(row=section_row, column=col * 2 + 1, sticky="w", padx=5, pady=5)
                        self.entries[key] = var

                    elif isinstance(value, int):  # Spinbox for integers
                        spinbox = tk.Spinbox(section_frame, from_=0, to=1000, increment=1, width=10)
                        spinbox.delete(0, "end")
                        spinbox.insert(0, str(value))
                        spinbox.grid(row=section_row, column=col * 2 + 1, sticky="w", padx=5, pady=5)
                        self.entries[key] = spinbox
                    
                    elif isinstance(value, float):  # Spinbox for integers
                        spinbox = tk.Spinbox(section_frame, from_=0, to=1, increment=0.1, width=10)
                        spinbox.delete(0, "end")
                        spinbox.insert(0, str(value))
                        spinbox.grid(row=section_row, column=col * 2 + 1, sticky="w", padx=5, pady=5)
                        self.entries[key] = spinbox

                    elif isinstance(value, list):  # Text widget for lists
                        text = tk.Text(section_frame, height=3, width=30, wrap="word")
                        text.insert("1.0", ",".join(map(str, value)))
                        text.grid(row=section_row, column=col * 2 + 1, sticky="w", padx=5, pady=5)
                        self.entries[key] = text

                    else:  # Default: Single-line text entry
                        entry = ttk.Entry(section_frame, width=40)
                        entry.insert(0, str(value))
                        entry.grid(row=section_row, column=col * 2 + 1, sticky="w", padx=5, pady=5)
                        self.entries[key] = entry

                    # Alternate between columns
                    if col == 1:  # If it's the second column, go to the next row
                        col = 0
                        section_row += 1
                    else:
                        col += 1
                
            else:
                section_row = 0  # Row tracker for each section
                for key in keys:
                    value = self.config_data.get(key, "")
                    label_text = descriptive_labels.get(key, key)

                    ttk.Label(section_frame, text=label_text).grid(row=section_row, column=0, sticky="w", padx=5, pady=5)

                    def on_save_frames_toggle():
                        if self.entries["save_frames"].get():  # Check if save_frames is True
                            self.entries["frames_to_save"].config(state="normal")
                        else:
                            self.entries["frames_to_save"].config(state="disabled")
                    
                    if key in ["video_source", "output_directory"]:  # Directory or file chooser
                        def browse_path(entry_widget, is_file=False):
                            if is_file:
                                selected_path = filedialog.askopenfilename()  # File chooser
                            else:
                                selected_path = filedialog.askdirectory()  # Directory chooser
                            
                            if selected_path:  # Update the entry only if a valid path is selected
                                entry_widget.delete(0, "end")
                                entry_widget.insert(0, selected_path)

                        entry = ttk.Entry(section_frame, width=40)
                        entry.insert(0, str(value))
                        entry.grid(row=section_row, column=1, sticky="w", padx=5, pady=5)
                        self.entries[key] = entry

                        if key == "video_source":  # Allow file or directory selection for video_source
                            browse_button = ttk.Button(
                                section_frame,
                                text="Browse",
                                command=lambda e=entry: browse_path(e, is_file=True)
                            )
                        elif key == "output_directory":  # Only allow directory selection for output_directory
                            browse_button = ttk.Button(
                                section_frame,
                                text="Browse",
                                command=lambda e=entry: browse_path(e, is_file=False)
                            )
                            
                        browse_button.grid(row=section_row, column=2, padx=5, pady=5)


                    elif key == "save_frames":  # Checkbox for Save Individual Frames
                        var = tk.BooleanVar(value=value)
                        chk = ttk.Checkbutton(section_frame, variable=var, command=on_save_frames_toggle)
                        chk.grid(row=section_row, column=1, sticky="w", padx=5, pady=5)
                        self.entries[key] = var

                    elif key == "frames_to_save":  # Spinbox for Number of Frames to Save
                        spinbox = tk.Spinbox(section_frame, from_=0, to=1000, increment=1, width=10)
                        spinbox.delete(0, "end")
                        spinbox.insert(0, str(value))
                        spinbox.grid(row=section_row, column=1, sticky="w", padx=5, pady=5)
                        spinbox.config(state="normal" if self.entries["save_frames"].get() else "disabled")
                        self.entries[key] = spinbox
                    
                    elif key == "background_transparency":
                        def validate_float(value_if_allowed):
                        # Allow only valid floats between 0 and 1
                            try:
                                value = float(value_if_allowed)
                                return 0.0 <= value <= 1.0
                            except ValueError:
                                return False

                        validate_command = section_frame.register(validate_float)

                        spinbox = tk.Spinbox(
                            section_frame, 
                            from_=0.0, 
                            to=1.0, 
                            increment=0.1, 
                            width=10, 
                            format="%.1f", 
                            validate="key", 
                            validatecommand=(validate_command, "%P")
                        )
                        spinbox.delete(0, "end")
                        spinbox.insert(0, str(value))
                        spinbox.grid(row=section_row, column=1, sticky="w", padx=5, pady=5)
                        self.entries[key] = spinbox

                    elif isinstance(value, bool):  # Checkbox for booleans
                        var = tk.BooleanVar(value=value)
                        chk = ttk.Checkbutton(section_frame, variable=var)
                        chk.grid(row=section_row, column=1, sticky="w", padx=5, pady=5)
                        self.entries[key] = var

                    elif isinstance(value, int):  # Spinbox for integers
                        spinbox = tk.Spinbox(section_frame, from_=0, to=1000, increment=1, width=10)
                        spinbox.delete(0, "end")
                        spinbox.insert(0, str(value))
                        spinbox.grid(row=section_row, column=1, sticky="w", padx=5, pady=5)
                        self.entries[key] = spinbox

                    


                    elif key == "camera_resolution":  # Dropdown for resolution
                        options = ["1920x1080", "1280x720"]
                        # Handle different formats gracefully
                        if isinstance(value, list) and len(value) == 2:
                            current_value = f"{value[0]}x{value[1]}"  # Convert list to string format for the dropdown
                        else:
                            current_value = "1920x1080"  # Default value if the format is incorrect or missing

                        combo = ttk.Combobox(section_frame, values=options, state="readonly", width=10)
                        combo.set(current_value)
                        combo.grid(row=section_row, column=1, sticky="w", padx=5, pady=5)
                        self.entries[key] = combo

                    elif key == "video_codec":  # Dropdown for fixed options
                        options = ["DIVX", "X264"]
                        combo = ttk.Combobox(section_frame, values=options, state="readonly", width=10)
                        combo.set(value)
                        combo.grid(row=section_row, column=1, sticky="w", padx=5, pady=5)
                        self.entries[key] = combo

                    elif isinstance(value, list):  # Text widget for lists
                        text = tk.Text(section_frame, height=3, width=30, wrap="word")
                        text.insert("1.0", ",".join(map(str, value)))
                        text.grid(row=section_row, column=4, sticky="w", padx=5, pady=5)
                        self.entries[key] = text

                    else:  # Default: Single-line text entry
                        entry = ttk.Entry(section_frame, width=40)
                        entry.insert(0, str(value))
                        entry.grid(row=section_row, column=1, sticky="w", padx=5, pady=5)
                        self.entries[key] = entry

                    section_row += 1

            row += 1  # Move to the next section






    def save_config(self):
        """Save configuration to JSON file."""
        try:
            for key, widget in self.entries.items():
                if isinstance(widget, tk.BooleanVar):
                    self.config_data[key] = widget.get()
                else:
                    value = widget.get()
                    if key == "camera_resolution":  # Convert resolution string to a list of integers
                        self.config_data[key] = list(map(int, value.split("x")))
                    elif key == "background_transparency": # Convert # Convert resolution string to a float
                        self.config_data[key] = float(value)
                    elif isinstance(self.config_data[key], int):
                        self.config_data[key] = int(value)
                    elif isinstance(self.config_data[key], float):
                        self.config_data[key] = float(value)
                    elif isinstance(self.config_data[key], list):
                        self.config_data[key] = [int(i) if i.isdigit() else i for i in value.split(",")]
                    else:
                        self.config_data[key] = value

            with open(self.config_file, "w") as f:
                json.dump(self.config_data, f, indent=4)

            messagebox.showinfo("Success", f"Configuration saved to {self.config_file}.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving the config: {e}")


    def run_compression(self):
        """Run the video processing script and display its output in real-time."""

        # Save the configuration before running the script
        self.save_config()

        def read_output(process, text_widget):
            """Read output from process and display it in the text widget."""
            for line in iter(process.stdout.readline, ""):
                text_widget.insert("end", line)
                text_widget.see("end")  # Auto-scroll to the bottom
            process.stdout.close()

        def execute_script():
            try:
                # Open a subprocess for the script
                process = subprocess.Popen(
                    ["python", os.path.join(os.path.dirname(__file__), "EcoMotionZip_lite.py")],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                # Create a new output window
                output_window = tk.Toplevel(self.root)
                output_window.title("Compression Script Output")
                output_window.geometry("600x400")

                # Create a scrollable text widget
                text_widget = tk.Text(output_window, wrap="word")
                text_widget.pack(expand=True, fill="both")

                # Start a thread to read and display the output in real-time
                threading.Thread(target=read_output, args=(process, text_widget), daemon=True).start()

                # Add a close button
                ttk.Button(output_window, text="Close", command=output_window.destroy).pack(pady=5)

                # Wait for the process to complete
                process.wait()

                if process.returncode == 0:
                    messagebox.showinfo("Success", "Compression script executed successfully!")
                else:
                    messagebox.showerror("Error", f"Script exited with return code {process.returncode}.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

        # Start the script execution in a separate thread
        threading.Thread(target=execute_script, daemon=True).start()



if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("700x650")
    app = ConfigEditorApp(root)

    # Make the Tkinter window active and focused
    root.update_idletasks()  # Ensures the window is fully initialized
    root.focus_force()       # Brings the window to the foreground

    root.mainloop()
