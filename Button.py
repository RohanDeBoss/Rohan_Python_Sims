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
        self.root.geometry("1760x770")  # Increased height for better graph display
        
        # Flag to track if the application is closing
        self.is_closing = False
        
        # Add escape key binding to close application
        self.root.bind('<Escape>', lambda e: self.close_application())
        self.root.protocol("WM_DELETE_WINDOW", self.close_application)

        # Game variables
        self.current_count = 0
        self.high_score = 0
        self.scores = []
        self.score_counter = Counter()
        self.avg_score = 0
        self.total_presses = 0
        self.auto_press_active = False
        self.auto_press_thread = None
        self.lock = threading.Lock()
        
        # UI update tracking
        self.ui_updater_id = None
        self.graph_updater_id = None
        
        # Main frame structure - adjust ratio to give more space to graph
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid columns with weight to prioritize graph space
        self.main_frame.columnconfigure(0, weight=1)  # Left panel
        self.main_frame.columnconfigure(1, weight=3)  # Graph panel (3x wider)
        self.main_frame.rowconfigure(0, weight=1)

        # Left panel (controls and stats)
        self.left_frame = tk.Frame(self.main_frame, padx=10, pady=10)
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        # Right panel (graph)
        self.right_frame = tk.Frame(self.main_frame, padx=10, pady=10)
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        self.right_frame.rowconfigure(1, weight=1)  # Make graph expandable
        self.right_frame.columnconfigure(0, weight=1)

        # Circular button - more compact
        self.canvas = tk.Canvas(self.left_frame, width=120, height=120, highlightthickness=0)
        self.canvas.pack(pady=10)
        self.button_bg = self.canvas.create_oval(10, 10, 110, 110, fill="#4CAF50", outline="#2E7D32", width=2)
        self.button_text = self.canvas.create_text(60, 60, text="0", font=("Arial", 22, "bold"), fill="white")
        self.canvas.tag_bind(self.button_bg, "<Button-1>", lambda e: self.press_button())
        self.canvas.tag_bind(self.button_text, "<Button-1>", lambda e: self.press_button())

        # Score display - compact layout with current game and total presses
        score_frame = tk.Frame(self.left_frame)
        score_frame.pack(pady=5, fill=tk.X)
        
        # Stats section - rearranged to remove redundant high score
        stats_frame = tk.Frame(score_frame)
        stats_frame.pack(fill=tk.X)
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.columnconfigure(1, weight=1)
        
        # Show Average instead of High Score
        tk.Label(stats_frame, text="Average:", font=("Arial", 11)).grid(row=0, column=0, padx=2, sticky=tk.W)
        self.avg_score_label = tk.Label(stats_frame, text="0.00", font=("Arial", 11, "bold"))
        self.avg_score_label.grid(row=0, column=1, padx=2, sticky=tk.W)
        
        # Total presses in own row
        total_frame = tk.Frame(self.left_frame)
        total_frame.pack(pady=5, fill=tk.X)
        tk.Label(total_frame, text="Total Button Presses:", font=("Arial", 11)).pack(side=tk.LEFT, padx=2)
        self.total_presses_label = tk.Label(total_frame, text="0", font=("Arial", 11, "bold"))
        self.total_presses_label.pack(side=tk.LEFT, padx=2)

        # Auto-press buttons - more compact horizontal layout
        auto_frame = tk.LabelFrame(self.left_frame, text="Auto Press Speed", font=("Arial", 10, "bold"))
        auto_frame.pack(pady=5, fill=tk.X)
        
        # Create a grid for buttons
        btn_frame = tk.Frame(auto_frame)
        btn_frame.pack(pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)
        btn_frame.columnconfigure(3, weight=1)
        
        speeds = [1, 100, 500, 2000, 10000, 20000, 50000]
        row, col = 0, 0
        for speed in speeds:
            btn = ttk.Button(btn_frame, text=f"{speed}/s", command=lambda s=speed: self.start_auto_press(s), width=8)
            btn.grid(row=row, column=col, padx=2, pady=2)
            col += 1
            if col > 2:  # 4 buttons per row
                col = 0
                row += 1
                
        stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_auto_press, width=8)
        stop_btn.grid(row=row, column=col, padx=2, pady=2)
        
        self.auto_status_label = tk.Label(self.left_frame, text="Auto Press: Off", font=("Arial", 10, "italic"))
        self.auto_status_label.pack(pady=2)

        # Add note about escape key
        escape_note = tk.Label(self.left_frame, text="Press ESC to close", font=("Arial", 9), fg="gray")
        escape_note.pack(pady=2)

        # Graph setup - Title in the frame, not on the plot
        tk.Label(self.right_frame, text="Score Distribution", font=("Arial", 14, "bold")).grid(row=0, column=0, pady=5, sticky=tk.W)
        
        # Create figure with bigger size optimized for the space
        self.fig, self.ax = plt.subplots(figsize=(13, 8))
        self.fig.patch.set_facecolor('#F0F0F0')
        self.ax.set_xlabel('Score Value', fontsize=10)
        self.ax.set_ylabel('Percentage (%)', fontsize=10)
        self.ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Place canvas in the grid with expand
        self.canvas_frame = tk.Frame(self.right_frame)
        self.canvas_frame.grid(row=1, column=0, sticky="nsew")
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas_graph.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Stats panel - more compact layout
        self.stats_frame = tk.Frame(self.right_frame)
        self.stats_frame.grid(row=2, column=0, pady=5, sticky=tk.EW)
        self.stat_labels = {}
        
        # Rearranged stats to better use the space since we removed high score from the left panel
        stats = [
            ('Games', 'Games: 0'),
            ('High', 'High: 0'),
            ('Avg', 'Avg: 0.00'),
            ('Median', 'Median: 0'),
            ('Mode', 'Mode: N/A'),
            ('ReachMax', 'P(≥ High): N/A'),
            ('ReachMaxGames', 'P(High in Games): N/A')
        ]
        
        # Use grid with 3 stats per row for better space usage
        row, col = 0, 0
        for i, (key, text) in enumerate(stats):
            lbl = tk.Label(self.stats_frame, text=text, font=("Arial", 10), bg="#f0f0f0", relief=tk.GROOVE, padx=5, pady=3)
            lbl.grid(row=row, column=col, padx=3, pady=2, sticky=tk.W+tk.E)
            self.stat_labels[key] = lbl
            col += 1
            if col > 2:  # 3 stats per row
                col = 0
                row += 1
        
        # Configure columns to distribute space
        for i in range(3):
            self.stats_frame.columnconfigure(i, weight=1)

        # Initialize graph
        self.bars = None
        self.cumulative_line = None
        self.avg_line = None
        self.high_line = None  # Added line for high score
        self.x_data = []
        self.y_data = []
        self.update_graph_initial()

        # Pre-calculate some common probabilities for faster access
        self._prob_cache = {}
        for i in range(1, 101):
            self._prob_cache[i] = self.theoretical_prob_reaching(i)
            
        # Start periodic updaters
        self.start_updaters()

    def close_application(self):
        """Safely close the application with proper cleanup."""
        # Set closing flag to prevent further UI updates
        self.is_closing = True
        
        # Stop auto-press thread
        self.stop_auto_press()
        
        # Cancel scheduled updaters
        if self.ui_updater_id:
            self.root.after_cancel(self.ui_updater_id)
        if self.graph_updater_id:
            self.root.after_cancel(self.graph_updater_id)
        
        # Delay destruction slightly to allow threads to stop
        self.root.after(100, self.root.destroy)

    def start_updaters(self):
        """Start periodic UI and graph updates with optimized timing."""
        def ui_updater():
            if not self.is_closing:
                self.update_ui()
                self.ui_updater_id = self.root.after(100, ui_updater)  # Every 0.1s
            
        def graph_updater():
            if not self.is_closing:
                # Only update graph if there are scores
                if self.scores:
                    self.update_graph()
                self.graph_updater_id = self.root.after(500, graph_updater)  # Every 0.5s
            
        self.ui_updater_id = self.root.after(100, ui_updater)
        self.graph_updater_id = self.root.after(1000, graph_updater)

    def update_graph_initial(self):
        """Initialize the graph with a placeholder message."""
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Play games to see your stats!', ha='center', va='center', transform=self.ax.transAxes, fontsize=12)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.fig.tight_layout()
        self.canvas_graph.draw()

    def click_logic(self):
        """Handle manual clicks with minimal overhead."""
        reset_occurred = False
        with self.lock:
            self.current_count += 1
            self.total_presses += 1
            if random.randint(1, 100) <= self.current_count:
                if self.current_count > self.high_score:
                    self.high_score = self.current_count
                self.scores.append(self.current_count)
                self.score_counter[self.current_count] += 1
                self.avg_score = sum(self.scores) / len(self.scores) if self.scores else 0
                self.current_count = 0
                reset_occurred = True
        return reset_occurred

    def press_button(self):
        """Handle manual button presses with immediate feedback."""
        if self.is_closing:
            return
        
        reset_occurred = self.click_logic()
        with self.lock:
            count = self.current_count
        
        try:
            self.canvas.itemconfig(self.button_text, text=str(count))
            if reset_occurred:
                self.flash_button()
        except tk.TclError:
            # Canvas might be destroyed if application is closing
            pass

    def flash_button(self):
        """Flash the button red briefly on reset."""
        if self.is_closing:
            return
            
        try:
            self.canvas.itemconfig(self.button_bg, fill="#e74c3c")  # Flash red
            self.root.after(200, lambda: self.safe_restore_button_color())  # Restore after 100ms
        except tk.TclError:
            # Canvas might be destroyed if application is closing
            pass

    def safe_restore_button_color(self):
        """Safely restore button to green with error handling."""
        if self.is_closing:
            return
            
        try:
            self.canvas.itemconfig(self.button_bg, fill="#4CAF50")  # Always restore to green
        except tk.TclError:
            # Canvas might be destroyed if application is closing
            pass

    def theoretical_prob_reaching(self, s):
        """Calculate the theoretical probability of reaching score s with caching."""
        if s in self._prob_cache:
            return self._prob_cache[s]
            
        if s <= 1:
            return 100.0
        prob = 1.0
        for k in range(1, s):
            prob *= (1 - k / 100)
        result = prob * 100
        self._prob_cache[s] = result
        return result
        
    def calculate_prob_in_n_games(self, score, num_games):
        """Calculate probability of reaching a score in a specific number of games."""
        # Probability of reaching the score in a single game
        p_single = self.theoretical_prob_reaching(score) / 100.0
        
        # Probability of reaching the score at least once in n games
        p_at_least_once = 1.0 - ((1.0 - p_single) ** num_games)
        
        return p_at_least_once * 100  # Return as percentage

    def update_ui(self):
        """Update UI elements periodically with extra decimal place."""
        if self.is_closing:
            return
            
        with self.lock:
            count = self.current_count
            total_presses = self.total_presses
            high_score = self.high_score
            avg_score = self.avg_score
            scores = self.scores[:]
            score_counter = self.score_counter.copy()
        
        try:
            self.canvas.itemconfig(self.button_text, text=str(count))
            self.total_presses_label.config(text=str(total_presses))
            # Only update average score in the left panel (high score removed)
            self.avg_score_label.config(text=f"{avg_score:.3f}")  # Extra decimal place
            
            if scores:
                median_score = sorted(scores)[len(scores)//2]
                most_common = score_counter.most_common(1)[0]
                prob_reach_max = self.theoretical_prob_reaching(high_score)
                
                # Calculate probability of reaching high score in this many games
                num_games = len(scores)
                prob_in_games = self.calculate_prob_in_n_games(high_score, num_games)
            else:
                median_score = 0
                most_common = (0, 0)
                prob_reach_max = 0
                prob_in_games = 0
                
            self.stat_labels['Games'].config(text=f"Games: {len(scores)}")
            self.stat_labels['Avg'].config(text=f"Avg: {avg_score:.3f}")  # Extra decimal place
            self.stat_labels['Median'].config(text=f"Median: {median_score}")
            self.stat_labels['Mode'].config(text=f"Mode: {most_common[0]} ({most_common[1]}x)")
            self.stat_labels['ReachMax'].config(text=f"P(≥ High): {prob_reach_max:.6f}%")  # 6 decimal places
            
            # Add new stat showing probability of achieving high score in the current number of games
            self.stat_labels['ReachMaxGames'].config(
                text=f"P(High in {len(scores)} games): {prob_in_games:.4f}%"
            )
            
        except tk.TclError:
            # Widgets might be destroyed if application is closing
            pass

    def update_graph(self):
        """Update the score distribution graph with performance optimizations."""
        if self.is_closing:
            return
            
        with self.lock:
            score_counter = self.score_counter.copy()
            scores = self.scores[:]
            avg_score = self.avg_score
            high_score = self.high_score
            
        if not scores:
            self.update_graph_initial()
            return
            
        if not score_counter:
            return
        
        try:
            sorted_items = sorted(score_counter.items())
            new_x = [item[0] for item in sorted_items]
            new_y = [item[1] for item in sorted_items]
            total_games = len(scores)
            percentages = [count / total_games * 100 for count in new_y]
            cumulative_percentages = [sum(new_y[i:]) / total_games * 100 for i in range(len(new_x))]

            structure_changed = new_x != self.x_data
            self.x_data = new_x
            self.y_data = percentages

            if self.bars is None or structure_changed:
                self.ax.clear()
                self.ax.set_xlabel('Score Value', fontsize=10)
                self.ax.set_ylabel('Percentage (%)', fontsize=10)
                self.ax.grid(True, linestyle='--', alpha=0.7, axis='y')
                norm = plt.Normalize(min(percentages) if percentages else 0, max(percentages) if percentages else 1)
                colors = plt.cm.viridis(norm(percentages))
                self.bars = self.ax.bar(new_x, percentages, color=colors, alpha=0.8, width=0.75)
                for i, bar in enumerate(self.bars):
                    height = bar.get_height()
                    count = new_y[i]
                    percentage = percentages[i]
                    # Conditional formatting based on percentage value
                    if percentage > 1.0:
                        label = f"{count}\n{percentage:.2f}%"  # 2 decimals for > 1%
                    elif percentage < 0.01:
                        label = f"{count}\n{percentage:.4f}%"  # 4 decimals for < 0.01%
                    else:
                        label = f"{count}\n{percentage:.3f}%"  # 3 decimals for 0.01% to 1%
                    self.ax.text(bar.get_x() + bar.get_width()/2., height, label, ha='center', va='bottom', fontsize=8)
                self.cumulative_line, = self.ax.plot(new_x, cumulative_percentages, color='red', marker='o', markersize=4, label='P(score ≥ s) (%)')
                self.avg_line = self.ax.axvline(x=avg_score, color='#e74c3c', linestyle='--', label=f'Avg: {avg_score:.3f}')
                self.high_line = self.ax.axvline(x=high_score, color='#2980b9', linestyle='-', linewidth=2, label=f'High: {high_score}')
                
                x_min = min(new_x) - 0.5
                x_max = max(max(new_x), high_score) + 1
                self.ax.set_xlim(x_min, x_max)
                self.ax.set_ylim(0, 105)
                self.ax.set_xticks(new_x)
                self.ax.legend(loc='upper right')
                self.fig.tight_layout()
                self.canvas_graph.draw()
            else:
                for i, (bar, percentage) in enumerate(zip(self.bars, percentages)):
                    bar.set_height(percentage)
                for txt in self.ax.texts:
                    txt.remove()
                for i, bar in enumerate(self.bars):
                    height = bar.get_height()
                    count = new_y[i]
                    percentage = percentages[i]
                    # Conditional formatting based on percentage value
                    if percentage >= 1.0:
                        label = f"{count}\n{percentage:.2f}%"  # 2 decimals for > 1%
                    elif percentage < 0.01:
                        label = f"{count}\n{percentage:.4f}%"  # 4 decimals for < 0.01%
                    else:
                        label = f"{count}\n{percentage:.3f}%"  # 3 decimals for 0.01% to 1%
                    self.ax.text(bar.get_x() + bar.get_width()/2., height, label, ha='center', va='bottom', fontsize=8)
                self.cumulative_line.set_data(new_x, cumulative_percentages)
                self.avg_line.set_xdata([avg_score, avg_score])
                self.avg_line.set_label(f'Avg: {avg_score:.3f}')
                if self.high_line:
                    self.high_line.set_xdata([high_score, high_score])
                    self.high_line.set_label(f'High: {high_score}')
                else:
                    self.high_line = self.ax.axvline(x=high_score, color='#2980b9', linestyle='-', linewidth=2, label=f'High: {high_score}')
                
                x_min = min(new_x) - 0.5
                x_max = max(max(new_x), high_score) + 1
                self.ax.set_xlim(x_min, x_max)
                self.ax.legend(loc='upper right')
                self.canvas_graph.draw_idle()
        except Exception as e:
            print(f"Error updating graph: {e}")

    def start_auto_press(self, speed):
        """Start the auto-press thread at the specified speed."""
        if self.is_closing:
            return
            
        if self.auto_press_active:
            self.stop_auto_press()
            
        self.auto_press_active = True
        try:
            self.auto_status_label.config(text=f"Auto Press: {speed}/s")
        except tk.TclError:
            # Widget might be destroyed if application is closing
            return
            
        # Start a new thread for auto-pressing
        self.auto_press_thread = threading.Thread(target=self.auto_press_loop, args=(speed,), daemon=True)
        self.auto_press_thread.start()

    def auto_press_loop(self, speed):
        """Run the auto-press simulation loop with batching for speeds >= 10000/s."""
        interval = 1.0 / speed if speed > 0 else 0.001  # Base interval per click
        last_ui_update = time.time()
        
        if speed >= 10000:
            # Batching for high speeds (>= 10000/s)
            batch_interval = 0.05  # Process 50ms worth of clicks per batch
            batch_size = max(1, int(batch_interval / interval))  # Number of clicks per batch
            actual_interval = max(interval, 0.0001)  # Minimum sleep to avoid CPU overload
            local_count = self.current_count  # Local counter for batching
            clicks_since_update = 0
            
            while self.auto_press_active and not self.is_closing:
                try:
                    # Process a batch of clicks
                    for _ in range(batch_size):
                        if not self.auto_press_active or self.is_closing:
                            break
                        local_count += 1
                        clicks_since_update += 1
                        if random.randint(1, 100) <= local_count:
                            with self.lock:
                                self.total_presses += clicks_since_update
                                if local_count > self.high_score:
                                    self.high_score = local_count
                                self.scores.append(local_count)
                                self.score_counter[local_count] += 1
                                self.avg_score = sum(self.scores) / len(self.scores) if self.scores else 0
                                self.current_count = 0
                            local_count = 0
                            clicks_since_update = 0
                            reset_occurred = True
                        else:
                            reset_occurred = False
                        
                        # Sleep per click within batch (minimal impact due to batching)
                        time.sleep(actual_interval)
                    
                    # Periodic UI update after batch
                    current_time = time.time()
                    if reset_occurred or (current_time - last_ui_update) > 0.1:
                        with self.lock:
                            self.current_count = local_count
                            self.total_presses += clicks_since_update
                            clicks_since_update = 0
                        if not self.is_closing:
                            try:
                                def safe_update():
                                    if not self.is_closing:
                                        try:
                                            self.canvas.itemconfig(self.button_text, text=str(local_count))
                                            if reset_occurred:
                                                self.flash_button()
                                        except tk.TclError:
                                            pass
                                self.root.after_idle(safe_update)
                            except tk.TclError:
                                break
                        last_ui_update = current_time
                
                except Exception as e:
                    if not self.is_closing:
                        print(f"Error in auto-press loop (batched): {e}")
                    time.sleep(0.08)
        else:
            # Original per-click logic for speeds < 10000/s
            actual_interval = max(interval, 0.0008)
            while self.auto_press_active and not self.is_closing:
                try:
                    reset_occurred = self.click_logic()
                    current_time = time.time()
                    if reset_occurred or (current_time - last_ui_update) > 0.1:
                        with self.lock:
                            count = self.current_count
                        if not self.is_closing:
                            try:
                                def safe_update():
                                    if not self.is_closing:
                                        try:
                                            self.canvas.itemconfig(self.button_text, text=str(count))
                                            if reset_occurred:
                                                self.flash_button()
                                        except tk.TclError:
                                            pass
                                self.root.after_idle(safe_update)
                            except tk.TclError:
                                break
                        last_ui_update = current_time
                    time.sleep(actual_interval)
                
                except Exception as e:
                    if not self.is_closing:
                        print(f"Error in auto-press loop: {e}")
                    time.sleep(0.08)
                
                # Check if we should stop
                if self.is_closing:
                    break

    def stop_auto_press(self):
        """Stop the auto-press thread."""
        self.auto_press_active = False
        try:
            self.auto_status_label.config(text="Auto Press: Off")
        except tk.TclError:
            # Widget might be destroyed if application is closing
            pass

if __name__ == "__main__":
    root = tk.Tk()
    game = ButtonGame(root)
    root.mainloop()