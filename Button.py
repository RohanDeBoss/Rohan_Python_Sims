import tkinter as tk
from tkinter import ttk
import random
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter
import numpy as np

class ButtonGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Button Game")
        self.root.geometry("900x600")  # Wider window
        
        # Game variables
        self.current_count = 0
        self.high_score = 0
        self.scores = []
        self.score_counter = Counter()  # For efficient counting
        self.avg_score = 0
        self.total_presses = 0
        self.auto_press_active = False
        self.auto_press_thread = None
        self.last_update_time = 0
        self.update_interval = 500  # Minimum ms between graph updates
        
        # Create main layout with two columns
        self.left_frame = tk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, padx=20, pady=10, fill=tk.BOTH)
        
        self.right_frame = tk.Frame(root)
        self.right_frame.pack(side=tk.RIGHT, padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        # Canvas for circular button (left frame)
        self.canvas = tk.Canvas(self.left_frame, width=150, height=150, highlightthickness=0)
        self.canvas.pack(pady=20)
        
        # Create circular button
        self.button_bg = self.canvas.create_oval(10, 10, 140, 140, fill="#4CAF50", outline="#2E7D32", width=2)
        self.button_text = self.canvas.create_text(75, 75, text="0", font=("Arial", 24, "bold"), fill="white")
        
        # Bind click event to canvas
        self.canvas.tag_bind(self.button_bg, "<Button-1>", lambda e: self.press_button())
        self.canvas.tag_bind(self.button_text, "<Button-1>", lambda e: self.press_button())
        
        # Score display frame
        score_frame = tk.Frame(self.left_frame)
        score_frame.pack(pady=15, fill=tk.X)
        
        tk.Label(score_frame, text="High Score:", font=("Arial", 12)).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.high_score_label = tk.Label(score_frame, text="0", font=("Arial", 12, "bold"))
        self.high_score_label.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        tk.Label(score_frame, text="Average Score:", font=("Arial", 12)).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.avg_score_label = tk.Label(score_frame, text="0.0", font=("Arial", 12, "bold"))
        self.avg_score_label.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Total presses display
        total_frame = tk.Frame(self.left_frame)
        total_frame.pack(pady=10, fill=tk.X)
        
        tk.Label(total_frame, text="Total Button Presses:", font=("Arial", 12)).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.total_presses_label = tk.Label(total_frame, text="0", font=("Arial", 12, "bold"))
        self.total_presses_label.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        # Auto-press buttons frame
        auto_frame = tk.LabelFrame(self.left_frame, text="Auto Press Options", font=("Arial", 10, "bold"))
        auto_frame.pack(pady=15, fill=tk.X)
        
        for i, speed in enumerate([10, 100, 1000]):
            tk.Button(
                auto_frame, 
                text=f"{speed}/s",
                bg="#3498db",
                fg="white",
                command=lambda s=speed: self.toggle_auto_press(s)
            ).grid(row=0, column=i, padx=10, pady=10, sticky=tk.W)
        
        self.auto_status_label = tk.Label(self.left_frame, text="Auto Press: Off", font=("Arial", 10, "italic"))
        self.auto_status_label.pack(pady=5)
        
        # Stop button (initially hidden)
        self.stop_button = tk.Button(
            self.left_frame,
            text="STOP Auto Press",
            bg="#e74c3c",
            fg="white",
            command=self.stop_auto_press
        )
        
        # Title for the graph
        tk.Label(self.right_frame, text="Score Distribution", font=("Arial", 14, "bold")).pack(pady=5, anchor=tk.W)
        
        # Create the matplotlib figure (right frame)
        self.fig, self.ax = plt.subplots(figsize=(7, 4.5))
        self.fig.patch.set_facecolor('#F0F0F0')  # Match Tkinter background
        self.ax.set_title('Score Frequency', fontsize=12)
        self.ax.set_xlabel('Score Value', fontsize=10)
        self.ax.set_ylabel('Frequency (Count)', fontsize=10)
        self.ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Setup the canvas - using double-buffered rendering can help with performance
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas_widget = self.canvas_graph.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Add stats display below graph
        self.stats_frame = tk.Frame(self.right_frame)
        self.stats_frame.pack(pady=10, fill=tk.X)
        
        # Create labels for stats in a grid (5 columns)
        self.stat_labels = {}
        stats = [
            ('Games', 'Games: 0'), 
            ('High', 'High: 0'), 
            ('Avg', 'Avg: 0.0'),
            ('Median', 'Median: 0'), 
            ('Mode', 'Mode: N/A')
        ]
        
        for i, (key, text) in enumerate(stats):
            lbl = tk.Label(self.stats_frame, text=text, font=("Arial", 10), 
                      bg="#f0f0f0", relief=tk.GROOVE, padx=5, pady=3)
            lbl.grid(row=0, column=i, padx=5, sticky=tk.W+tk.E)
            self.stat_labels[key] = lbl
            self.stats_frame.grid_columnconfigure(i, weight=1)
        
        # Pre-render empty graph
        self.update_graph_initial()
        
        # Create bars collection for efficient updates
        self.bars = None
        self.x_data = []
        self.y_data = []
        
    def update_graph_initial(self):
        """Initial graph setup with placeholder text."""
        self.ax.text(0.5, 0.5, 'Play games to see your stats!', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=self.ax.transAxes, fontsize=12)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.fig.tight_layout()
        self.canvas_graph.draw()
    
    def press_button(self):
        # Increment counters
        self.current_count += 1
        self.total_presses += 1
        
        # Update UI
        self.canvas.itemconfig(self.button_text, text=str(self.current_count))
        self.total_presses_label.config(text=str(self.total_presses))
        
        # Check if reset occurs based on probability
        if random.randint(1, 100) <= self.current_count:
            # Game over - update stats
            if self.current_count > self.high_score:
                self.high_score = self.current_count
                self.high_score_label.config(text=str(self.high_score))
            
            # Update score collection
            self.scores.append(self.current_count)
            self.score_counter[self.current_count] += 1
            
            # Calculate average
            self.avg_score = sum(self.scores) / len(self.scores)
            self.avg_score_label.config(text=f"{self.avg_score:.1f}")
            
            # Reset counter
            self.current_count = 0
            self.canvas.itemconfig(self.button_text, text="0")
            
            # Flash the button to indicate reset
            self.flash_button()
            
            # Update stats panel immediately
            self.update_stats_panel()
            
            # Check if enough time has passed for graph update
            current_time = time.time() * 1000  # Convert to ms
            if current_time - self.last_update_time > self.update_interval:
                self.update_graph(force_redraw=True)
                self.last_update_time = current_time
            else:
                # Schedule a deferred update 
                self.root.after(self.update_interval, lambda: self.update_graph(force_redraw=False))
    
    def update_stats_panel(self):
        """Update just the statistics labels without redrawing the graph."""
        # Calculate stats
        median_score = sorted(self.scores)[len(self.scores)//2] if self.scores else 0
        most_common = self.score_counter.most_common(1)[0] if self.scores else (0, 0)
        
        # Update labels
        self.stat_labels['Games'].config(text=f"Games: {len(self.scores)}")
        self.stat_labels['High'].config(text=f"High: {self.high_score}")
        self.stat_labels['Avg'].config(text=f"Avg: {self.avg_score:.1f}")
        self.stat_labels['Median'].config(text=f"Median: {median_score}")
        self.stat_labels['Mode'].config(text=f"Mode: {most_common[0]} ({most_common[1]}x)")
    
    def update_graph(self, force_redraw=False):
        """Update the graph with current score distribution."""
        try:
            if not self.scores:
                return
            
            # Get sorted score counts
            sorted_items = sorted(self.score_counter.items())
            new_x = [item[0] for item in sorted_items]
            new_y = [item[1] for item in sorted_items]
            
            # Check if data structure changed (requiring full redraw)
            structure_changed = new_x != self.x_data
            
            # Store current data
            self.x_data = new_x
            self.y_data = new_y
            
            # If first time or structure changed, do a full redraw
            if self.bars is None or structure_changed or force_redraw:
                self.ax.clear()
                self.ax.set_title('Score Frequency', fontsize=12)
                self.ax.set_xlabel('Score Value', fontsize=10)
                self.ax.set_ylabel('Frequency (Count)', fontsize=10)
                self.ax.grid(True, linestyle='--', alpha=0.7, axis='y')
                
                # For color gradient
                norm = plt.Normalize(min(new_y), max(new_y))
                colors = plt.cm.viridis(norm(new_y))
                
                # Create new bars
                self.bars = self.ax.bar(new_x, new_y, color=colors, alpha=0.8, width=0.8)
                
                # Add value labels
                for bar in self.bars:
                    height = bar.get_height()
                    self.ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom', fontsize=9)
                
                # Add average line
                self.avg_line = self.ax.axvline(x=self.avg_score, color='#e74c3c', 
                                        linestyle='--', label=f'Avg: {self.avg_score:.1f}')
                
                # Set limits
                max_y = max(new_y)
                self.ax.set_ylim(0, max(5, max_y * 1.2))
                self.ax.set_xlim(min(new_x) - 0.5, max(new_x) + 0.5)
                
                # Set x-axis ticks
                self.ax.set_xticks(new_x)
                
                # Add legend
                self.ax.legend(loc='upper right')
                
                # Adjust layout
                self.fig.tight_layout()
                
                # Full redraw
                self.canvas_graph.draw()
            else:
                # Efficient update - just update the heights and average line
                for bar, new_height in zip(self.bars, new_y):
                    bar.set_height(new_height)
                
                # Update average line
                self.avg_line.set_xdata([self.avg_score, self.avg_score])
                self.avg_line.set_label(f'Avg: {self.avg_score:.1f}')
                
                # Update legend
                self.ax.legend(loc='upper right')
                
                # Use blit for faster rendering (only update changed parts)
                self.canvas_graph.draw_idle()
                
        except Exception as e:
            print(f"Error updating graph: {e}")
    
    def flash_button(self):
        original_fill = self.canvas.itemcget(self.button_bg, "fill")
        self.canvas.itemconfig(self.button_bg, fill="#e74c3c")  # Red color
        self.root.after(200, lambda: self.canvas.itemconfig(self.button_bg, fill=original_fill))
    
    def toggle_auto_press(self, clicks_per_second):
        if self.auto_press_active:
            self.stop_auto_press()
        else:
            self.auto_press_active = True
            self.auto_status_label.config(text=f"Auto Press: {clicks_per_second}/s")
            self.stop_button.pack(pady=5)
            
            # Limit actual click rate to prevent overwhelming the UI
            effective_rate = min(clicks_per_second, 50)
            delay = 1.0 / effective_rate
            
            # Adjust graph update interval based on click speed
            if clicks_per_second > 50:
                self.update_interval = 1000  # ms between updates for fast clicking
            
            self.auto_press_thread = threading.Thread(
                target=self.auto_press_loop,
                args=(delay, clicks_per_second, effective_rate),
                daemon=True
            )
            self.auto_press_thread.start()
    
    def auto_press_loop(self, delay, requested_rate, actual_rate):
        click_batch = max(1, round(requested_rate / actual_rate))
        
        while self.auto_press_active:
            for _ in range(click_batch):
                if not self.auto_press_active:
                    break
                self.root.after_idle(self.press_button)
            time.sleep(delay)
    
    def stop_auto_press(self):
        self.auto_press_active = False
        self.auto_status_label.config(text="Auto Press: Off")
        self.stop_button.pack_forget()
        self.update_interval = 500  # Reset to default update interval

if __name__ == "__main__":
    root = tk.Tk()
    game = ButtonGame(root)
    root.mainloop()