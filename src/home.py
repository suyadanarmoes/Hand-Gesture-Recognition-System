import tkinter as tk
from PIL import Image, ImageTk
import subprocess
import sys
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

BG_COLOR = "#dff3ff"
CARD_COLOR = "#ffffff"
NAVY = "#08245c"
ACCENT = "#0d6efd"
MUTED = "#42526e"

# Define the function that will run when the Start button is clicked
def start_button_click():
    print("Start button clicked!")
    root.destroy()
    subprocess.Popen([sys.executable, str(BASE_DIR / "src" / "main.py")], cwd=BASE_DIR)  # This will run main.py as a separate process

def warm_up_model():
    try:
        from model_utils import load_or_train_model
        load_or_train_model()
        root.after(0, lambda: status_label.config(text="Model is ready. Click Start Recognition."))
    except Exception as error:
        root.after(0, lambda: status_label.config(text=f"Model warm-up failed: {error}"))

# Create the main window
root = tk.Tk()
root.title("Hand Gesture Recognition GUI")
root.geometry("800x600")
root.minsize(760, 560)
root.configure(bg=BG_COLOR)

# Load images
icon1_img = Image.open(BASE_DIR / "images" / "icon1.png")
icon1_img = icon1_img.resize((80, 80))  # Resize to fit the space
icon1_photo = ImageTk.PhotoImage(icon1_img)

icon2_img = Image.open(BASE_DIR / "images" / "icon2.png")
icon2_img = icon2_img.resize((80, 80))  # Resize to fit the space
icon2_photo = ImageTk.PhotoImage(icon2_img)

main_frame = tk.Frame(root, bg=CARD_COLOR, highlightbackground="#9cc9e8", highlightthickness=2)
main_frame.place(relx=0.5, rely=0.5, relwidth=0.88, relheight=0.84, anchor="center")

header_frame = tk.Frame(main_frame, bg=CARD_COLOR)
header_frame.pack(fill="x", padx=30, pady=(24, 10))

content_frame = tk.Frame(main_frame, bg=CARD_COLOR)
content_frame.pack(expand=True, fill="both", padx=36)

footer_frame = tk.Frame(main_frame, bg=CARD_COLOR)
footer_frame.pack(fill="x", padx=30, pady=(8, 26))

# Create labels for text
university_label = tk.Label(header_frame, text="Technological University (Meiktila)", bg=CARD_COLOR, fg=NAVY, font=("Sagona Book", 20, "bold"))
department_label = tk.Label(header_frame, text="Department of Information Technology", bg=CARD_COLOR, fg=MUTED, font=("Sagona Book", 16))
title_label1 = tk.Label(content_frame, text="Hand Gesture Recognition System", bg=CARD_COLOR, fg=NAVY, font=("Times New Roman", 28, "bold"))
title_label2 = tk.Label(content_frame, text="for English Alphabet", bg=CARD_COLOR, fg=ACCENT, font=("Times New Roman", 22, "bold"))
instruction_label = tk.Label(content_frame, text="After clicking Start, the webcam will open. Press ESC to close the webcam.", bg=CARD_COLOR, fg=MUTED, font=("Sagona Book", 14))
status_label = tk.Label(content_frame, text="Preparing recognition model...", bg=CARD_COLOR, fg=ACCENT, font=("Sagona Book", 12))
supervisor_label1 = tk.Label(footer_frame, text="Supervised By:", bg=CARD_COLOR, fg=NAVY, font=("Sagona Book", 13, "bold"))
supervisor_label2 = tk.Label(footer_frame, text="Dr. Thidar Khaing", bg=CARD_COLOR, fg=MUTED, font=("Sagona Book", 13))
author_label1 = tk.Label(footer_frame, text="Presented By:", bg=CARD_COLOR, fg=NAVY, font=("Sagona Book", 13, "bold"))
author_label2 = tk.Label(footer_frame, text="Ma Su Yadanar Moe (VI.IT-8)", bg=CARD_COLOR, fg=MUTED, font=("Sagona Book", 13))

# Create labels for images
icon1_label = tk.Label(header_frame, image=icon1_photo, bg=CARD_COLOR)
icon2_label = tk.Label(header_frame, image=icon2_photo, bg=CARD_COLOR)

# Create the start button
start_button = tk.Button(content_frame, text="Start Recognition", font=("Times New Roman", 15, "bold"), bg=NAVY, fg="white", activebackground=ACCENT, activeforeground="white", width=18, height=1, bd=0, cursor="hand2", command=start_button_click)

# Arrange widgets
icon1_label.pack(side="left")
icon2_label.pack(side="right")
university_label.pack(pady=(6, 2))
department_label.pack()

title_label1.pack(pady=(84, 4))
title_label2.pack()
instruction_label.pack(pady=(28, 18))
status_label.pack(pady=(0, 16))
start_button.pack()

supervisor_frame = tk.Frame(footer_frame, bg=CARD_COLOR)
supervisor_frame.pack(side="left", anchor="sw")
author_frame = tk.Frame(footer_frame, bg=CARD_COLOR)
author_frame.pack(side="right", anchor="se")

supervisor_label1.pack(in_=supervisor_frame, anchor="w")
supervisor_label2.pack(in_=supervisor_frame, anchor="w")
author_label1.pack(in_=author_frame, anchor="e")
author_label2.pack(in_=author_frame, anchor="e")

threading.Thread(target=warm_up_model, daemon=True).start()

# Run the main event loop
root.mainloop()
