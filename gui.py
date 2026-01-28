import tkinter as tk
from PIL import ImageTk
from decoder import PRESETS, latent_to_image

latent_values = [0.0] * 6
animation_steps = 40
animation_delay = 15

root = tk.Tk()
root.title("VAE Latent Space Explorer")

image_label = tk.Label(root)
image_label.grid(row=0, column=1, rowspan=8, padx=10, pady=10)

def update_display():
    img = latent_to_image(latent_values).resize((336, 336))
    tk_img = ImageTk.PhotoImage(img)
    image_label.img = tk_img
    image_label.config(image=tk_img)

sliders = []

def slider_changed(idx, val):
    latent_values[idx] = float(val)
    update_display()

for i in range(6):
    s = tk.Scale(
        root,
        from_=-3, to=3,
        resolution=0.0001,
        length=420,
        orient=tk.HORIZONTAL,
        label=f"Latent dim {i+1}",
        command=lambda v, i=i: slider_changed(i, v)
    )
    s.set(0)
    s.grid(row=i, column=0, padx=6, pady=4)
    sliders.append(s)

def animate_to(target):
    start = latent_values.copy()

    def step(frame):
        if frame > animation_steps:
            return
        t = frame / animation_steps
        for i in range(6):
            val = (1 - t) * start[i] + t * target[i]
            sliders[i].set(val)
            latent_values[i] = val
        update_display()
        root.after(animation_delay, lambda: step(frame + 1))

    step(0)

button_frame = tk.Frame(root)
button_frame.grid(row=6, column=0, pady=8)

for d in range(10):
    tk.Button(
        button_frame,
        text=f"Generate {d}",
        width=12,
        command=lambda d=d: animate_to(PRESETS[d])
    ).grid(row=d // 5, column=d % 5, padx=4, pady=4)

tk.Button(
    root,
    text="Reset All Sliders",
    width=22,
    bg="lightblue",
    command=lambda: animate_to([0.0]*6)
).grid(row=7, column=0, pady=6)

update_display()
root.mainloop()
