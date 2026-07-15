import tkinter as tk

root = tk.Tk()
root.title("Drag me over canvas")
root.attributes('-alpha', 0.3)
root.geometry("800x600+100+100")
root.wm_attributes("-topmost", 1)

def print_geom():
    print(root.winfo_x(), root.winfo_y(), root.winfo_width(), root.winfo_height())
    root.after(1000, print_geom)

root.after(1000, print_geom)
root.mainloop()
