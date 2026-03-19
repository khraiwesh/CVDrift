import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import random
import os
import copy
from datetime import datetime, timedelta
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.conversion.log import converter as log_converter


# Default configuration for activities with resource probabilities
# Each activity has 2 resources, starting with first resource at 100%
DEFAULT_ACTIVITIES_CONFIG = {
    "A": {
        "name": "Patient Registration",
        "resources": {
            "Receptionist 1": 1.0,
            "Receptionist 2": 0.0
        },
        "mean_duration": 30
    },
    "B": {
        "name": "Lab Test",
        "resources": {
            "Lab Technician 1": 1.0,
            "Lab Technician 2": 0.0
        },
        "mean_duration": 40
    },
    "C": {
        "name": "X-Ray Scan",
        "resources": {
            "Radiologist 1": 1.0,
            "Radiologist 2": 0.0
        },
        "mean_duration": 40
    },
    "D": {
        "name": "Diagnosis",
        "resources": {
            "Doctor 1": 1.0,
            "Doctor 2": 0.0
        },
        "mean_duration": 55
    },
    "E": {
        "name": "Treatment Evaluation",
        "resources": {
            "Nurse 1": 1.0,
            "Nurse 2": 0.0
        },
        "mean_duration": 45
    },
}

DEFAULT_XOR_PROBABILITY = 0.5
DEFAULT_CASE_GAP = 10.0


WORK_START_HOUR = 8   # 08:00
WORK_END_HOUR = 15    # 15:00


def advance_to_working_hours(dt: datetime) -> datetime:
    """If datetime is outside working hours (08:00-15:00), advance to next working day 08:00."""
    if dt.hour < WORK_START_HOUR:
        return dt.replace(hour=WORK_START_HOUR, minute=0, second=0, microsecond=0)
    elif dt.hour >= WORK_END_HOUR:
        next_day = dt + timedelta(days=1)
        return next_day.replace(hour=WORK_START_HOUR, minute=0, second=0, microsecond=0)
    return dt


def select_resource_weighted(resources_dict: dict) -> str:
    """Select a resource based on weighted probabilities."""
    resources = list(resources_dict.keys())
    weights = list(resources_dict.values())
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    else:
        weights = [1.0 / len(resources)] * len(resources)
    return random.choices(resources, weights=weights, k=1)[0]


def get_execution_time(mean_duration: float, std_ratio: float = 0.2) -> float:
    """Generate realistic execution time using normal distribution."""
    std_dev = mean_duration * std_ratio
    duration = np.random.normal(mean_duration, std_dev)
    return max(1.0, duration)


def create_event(case_id: str, activity_key: str, start_time: datetime, config: dict) -> tuple:
    """Create start and complete events for an activity."""
    activity_config = config[activity_key]
    activity_name = activity_config["name"]
    resource = select_resource_weighted(activity_config["resources"])
    duration = get_execution_time(activity_config["mean_duration"])
    complete_time = start_time + timedelta(minutes=duration)
    
    start_event = Event()
    start_event["concept:name"] = activity_name
    start_event["time:timestamp"] = start_time
    start_event["org:resource"] = resource
    start_event["lifecycle:transition"] = "start"
    
    complete_event = Event()
    complete_event["concept:name"] = activity_name
    complete_event["time:timestamp"] = complete_time
    complete_event["org:resource"] = resource
    complete_event["lifecycle:transition"] = "complete"
    
    return start_event, complete_event, complete_time


def generate_trace(case_id: str, start_time: datetime, config: dict, xor_probability: float) -> tuple:
    """Generate a single trace following the process model."""
    trace = Trace()
    trace.attributes["concept:name"] = case_id
    
    events = []
    current_time = start_time
    
    # Activity A
    start_evt, complete_evt, current_time = create_event(case_id, "A", current_time, config)
    events.extend([start_evt, complete_evt])
    
    # XOR Gateway
    xor_activity = "B" if random.random() < xor_probability else "C"
    current_time += timedelta(minutes=random.uniform(1, 3))
    
    start_evt, complete_evt, current_time = create_event(case_id, xor_activity, current_time, config)
    events.extend([start_evt, complete_evt])
    
    # Activity D
    current_time += timedelta(minutes=random.uniform(1, 3))
    start_evt, complete_evt, current_time = create_event(case_id, "D", current_time, config)
    events.extend([start_evt, complete_evt])
    
    # Activity E
    current_time += timedelta(minutes=random.uniform(1, 3))
    start_evt, complete_evt, current_time = create_event(case_id, "E", current_time, config)
    events.extend([start_evt, complete_evt])
    
    for evt in events:
        trace.append(evt)
    
    return trace, current_time


def generate_event_log(num_traces: int, start_time: datetime, config: dict, 
                       start_case_id: int, xor_probability: float,
                       min_gap: float, max_gap: float) -> tuple:
    """Generate an event log with multiple traces."""
    log = EventLog()
    current_time = start_time
    
    for i in range(num_traces):
        case_id = str(start_case_id + i)
        current_time = advance_to_working_hours(current_time)
        
        # If the case would exceed working hours, move to next working day
        while True:
            trace, end_time = generate_trace(case_id, current_time, config, xor_probability)
            end_within_day = (end_time.date() == current_time.date() and
                              end_time.hour < WORK_END_HOUR)
            if end_within_day:
                break
            # Shift entire case to next working day 08:00
            current_time = (current_time + timedelta(days=1)).replace(
                hour=WORK_START_HOUR, minute=0, second=0, microsecond=0
            )
        
        log.append(trace)
        gap = random.uniform(min_gap, max_gap)
        current_time = end_time + timedelta(minutes=gap)
    
    return log, current_time, start_case_id + num_traces


def merge_event_logs(log1: EventLog, log2: EventLog) -> EventLog:
    """Merge two event logs into one."""
    merged_log = EventLog()
    for trace in log1:
        merged_log.append(trace)
    for trace in log2:
        merged_log.append(trace)
    return merged_log


class SegmentFrame(ttk.LabelFrame):
    """Frame for configuring a single segment."""
    
    def __init__(self, parent, segment_num, is_before_drift=False, prev_config=None):
        title = "Segment 0 - Before Drift" if is_before_drift else f"Segment {segment_num} - Drift"
        super().__init__(parent, text=title, padding=10)
        
        self.segment_num = segment_num
        self.is_before_drift = is_before_drift
        self.prev_config = prev_config  # Previous segment's config for reference
        self.duration_vars = {}
        self.resource_vars = {}
        self.info_labels = {}  # Store info labels for updating
        
        self.create_widgets()
    
    def create_info_text(self, default_val, prev_val=None):
        """Create info text showing default and previous values."""
        if prev_val is not None and prev_val != default_val:
            return f"(default: {default_val}, prev: {prev_val})"
        return f"(default: {default_val})"
    
    def create_widgets(self):
        # Main container with grid
        row = 0
        
        # Number of traces - now with random option
        traces_frame = ttk.Frame(self)
        traces_frame.grid(row=row, column=0, columnspan=4, sticky="w", pady=2)
        ttk.Label(traces_frame, text="Number of Traces:").pack(side="left")
        
        # Random traces from predefined values
        trace_options = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        random_traces = random.choice(trace_options)
        default_traces = "100" if self.is_before_drift else str(random_traces)
        prev_traces = str(self.prev_config['traces']) if self.prev_config else None
        
        # For drift segments, always pick random
        if not self.is_before_drift:
            self.traces_var = tk.StringVar(value=str(random_traces))
        else:
            self.traces_var = tk.StringVar(value=prev_traces or default_traces)
        
        ttk.Entry(traces_frame, textvariable=self.traces_var, width=10).pack(side="left", padx=5)
        info_text = "(random 50-100)" if not self.is_before_drift else "(default: 100)"
        ttk.Label(traces_frame, text=info_text, font=("", 8), foreground="gray").pack(side="left")
        row += 1
        
        # XOR Probability
        xor_frame = ttk.Frame(self)
        xor_frame.grid(row=row, column=0, columnspan=4, sticky="w", pady=2)
        ttk.Label(xor_frame, text="Lab Test Probability (0-1):").pack(side="left")
        default_xor = "0.5"
        prev_xor = f"{self.prev_config['xor_prob']:.2f}" if self.prev_config else None
        self.xor_prob_var = tk.StringVar(value=prev_xor or default_xor)
        ttk.Entry(xor_frame, textvariable=self.xor_prob_var, width=10).pack(side="left", padx=5)
        info = self.create_info_text(default_xor, prev_xor)
        ttk.Label(xor_frame, text=info, font=("", 8), foreground="gray").pack(side="left")
        row += 1
        
        # Case Gap
        gap_frame = ttk.Frame(self)
        gap_frame.grid(row=row, column=0, columnspan=4, sticky="w", pady=2)
        ttk.Label(gap_frame, text="Case Gap (min):").pack(side="left")
        default_gap = "10"
        prev_gap = str(self.prev_config['case_gap']) if self.prev_config else None
        self.case_gap_var = tk.StringVar(value=prev_gap or default_gap)
        ttk.Entry(gap_frame, textvariable=self.case_gap_var, width=10).pack(side="left", padx=5)
        info = self.create_info_text(default_gap, prev_gap)
        ttk.Label(gap_frame, text=info + " ±20% variation", font=("", 8), foreground="gray").pack(side="left")
        row += 1
        
        # Separator
        ttk.Separator(self, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=10)
        row += 1
        
        # Activity Durations Header
        ttk.Label(self, text="Activity Durations (min)", font=("", 9, "bold")).grid(row=row, column=0, columnspan=4, sticky="w", pady=5)
        row += 1
        
        # Activity Duration inputs
        for activity_key, activity_data in DEFAULT_ACTIVITIES_CONFIG.items():
            dur_frame = ttk.Frame(self)
            dur_frame.grid(row=row, column=0, columnspan=4, sticky="w", pady=1)
            
            ttk.Label(dur_frame, text=f"{activity_key} - {activity_data['name']}:").pack(side="left")
            
            default_dur = activity_data['mean_duration']
            prev_dur = None
            if self.prev_config:
                prev_dur = self.prev_config['config'][activity_key]['mean_duration']
            
            var = tk.StringVar(value=str(prev_dur if prev_dur else default_dur))
            self.duration_vars[activity_key] = var
            ttk.Entry(dur_frame, textvariable=var, width=10).pack(side="left", padx=5)
            
            info = self.create_info_text(default_dur, prev_dur)
            ttk.Label(dur_frame, text=info, font=("", 8), foreground="gray").pack(side="left")
            row += 1
        
        # Separator
        ttk.Separator(self, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=10)
        row += 1
        
        # Resource Percentages Header
        ttk.Label(self, text="Resource Percentages (%)", font=("", 9, "bold")).grid(row=row, column=0, columnspan=4, sticky="w", pady=5)
        row += 1
        
        # Resource percentage inputs
        for activity_key, activity_data in DEFAULT_ACTIVITIES_CONFIG.items():
            # Activity label
            ttk.Label(self, text=f"{activity_key} - {activity_data['name']}:", font=("", 8, "bold")).grid(row=row, column=0, columnspan=4, sticky="w", pady=2)
            row += 1
            
            self.resource_vars[activity_key] = {}
            resources = list(activity_data['resources'].keys())
            
            for resource in resources:
                res_frame = ttk.Frame(self)
                res_frame.grid(row=row, column=0, columnspan=4, sticky="w", pady=1, padx=10)
                
                ttk.Label(res_frame, text=f"{resource}:", font=("", 8)).pack(side="left")
                
                default_pct = activity_data['resources'][resource] * 100
                prev_pct = None
                if self.prev_config:
                    prev_pct = self.prev_config['config'][activity_key]['resources'][resource] * 100
                
                var = tk.StringVar(value=f"{prev_pct:.1f}" if prev_pct is not None else f"{default_pct:.1f}")
                self.resource_vars[activity_key][resource] = var
                ttk.Entry(res_frame, textvariable=var, width=8).pack(side="left", padx=5)
                
                info = self.create_info_text(f"{default_pct:.1f}", f"{prev_pct:.1f}" if prev_pct is not None else None)
                ttk.Label(res_frame, text=info, font=("", 8), foreground="gray").pack(side="left")
                row += 1
    
    def get_config(self) -> dict:
        """Get configuration from this segment."""
        config = copy.deepcopy(DEFAULT_ACTIVITIES_CONFIG)
        
        # Update durations
        for activity_key, var in self.duration_vars.items():
            try:
                config[activity_key]['mean_duration'] = float(var.get())
            except ValueError:
                pass
        
        # Update resource percentages
        for activity_key, resources in self.resource_vars.items():
            total = 0
            for resource, var in resources.items():
                try:
                    pct = float(var.get())
                    config[activity_key]['resources'][resource] = pct / 100.0
                    total += pct
                except ValueError:
                    pass
            
            # Normalize if total != 100
            if total > 0 and abs(total - 100) > 0.1:
                for resource in config[activity_key]['resources']:
                    config[activity_key]['resources'][resource] /= (total / 100)
        
        # Calculate min/max from single case gap value (±20% variation)
        case_gap = float(self.case_gap_var.get() or 10)
        min_gap = case_gap * 0.8
        max_gap = case_gap * 1.2
        
        return {
            'traces': int(self.traces_var.get() or 100),
            'xor_prob': float(self.xor_prob_var.get() or 0.5),
            'case_gap': case_gap,
            'min_gap': min_gap,
            'max_gap': max_gap,
            'config': config
        }


class EventLogGeneratorApp:
    """Main application class."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Concept Drift Event Log Generator")
        self.root.geometry("1000x900")
        
        self.segments = []
        self.increment_vars = {}  # Store increment settings
        
        self.create_widgets()
    
    def create_widgets(self):
        # Main container with scrollbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Canvas for scrolling
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(self.scrollable_frame, text="Concept Drift Event Log Generator", 
                                font=("", 14, "bold"))
        title_label.pack(pady=10)
        
        subtitle = ttk.Label(self.scrollable_frame, 
                            text="Hospital Patient Admission Process\nA → (XOR: B/C) → D → E",
                            font=("", 10))
        subtitle.pack(pady=5)
        
        # ============ AUTO-INCREMENT SETTINGS ============
        increment_frame = ttk.LabelFrame(self.scrollable_frame, text="🔄 Auto-Increment Settings (Per Segment)", padding=10)
        increment_frame.pack(fill="x", padx=10, pady=10)
        
        # Explanation
        ttk.Label(increment_frame, text="Select which activities to apply auto-increment and set the increment values:", 
                  font=("", 8), foreground="gray").grid(row=0, column=0, columnspan=8, sticky="w", pady=5)
        
        row = 1
        
        # ---- Activity Duration Section ----
        ttk.Label(increment_frame, text="Activity Duration:", font=("", 9, "bold")).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.increment_vars['duration'] = tk.StringVar(value="5")
        ttk.Entry(increment_frame, textvariable=self.increment_vars['duration'], width=6).grid(row=row, column=1, sticky="w")
        ttk.Label(increment_frame, text="min/seg", font=("", 8), foreground="gray").grid(row=row, column=2, sticky="w")
        
        # Checkboxes for which activities to apply duration increment
        self.duration_checkboxes = {}
        col = 3
        for activity_key in DEFAULT_ACTIVITIES_CONFIG.keys():
            var = tk.BooleanVar(value=True)  # Default all selected
            self.duration_checkboxes[activity_key] = var
            ttk.Checkbutton(increment_frame, text=activity_key, variable=var).grid(row=row, column=col, sticky="w", padx=2)
            col += 1
        row += 1
        
        # ---- Routing Probability Section ----
        ttk.Label(increment_frame, text="Routing Prob:", font=("", 9, "bold")).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.increment_vars['xor_prob'] = tk.StringVar(value="10")
        ttk.Entry(increment_frame, textvariable=self.increment_vars['xor_prob'], width=6).grid(row=row, column=1, sticky="w")
        ttk.Label(increment_frame, text="%/seg", font=("", 8), foreground="gray").grid(row=row, column=2, sticky="w")
        ttk.Label(increment_frame, text="(Lab Test probability B vs C)", font=("", 8), foreground="gray").grid(row=row, column=3, columnspan=5, sticky="w")
        row += 1
        
        # ---- Arrival Time Section ----
        ttk.Label(increment_frame, text="Arrival Time:", font=("", 9, "bold")).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.increment_vars['case_gap'] = tk.StringVar(value="5")
        ttk.Entry(increment_frame, textvariable=self.increment_vars['case_gap'], width=6).grid(row=row, column=1, sticky="w")
        ttk.Label(increment_frame, text="min/seg", font=("", 8), foreground="gray").grid(row=row, column=2, sticky="w")
        ttk.Label(increment_frame, text="(Case gap between traces)", font=("", 8), foreground="gray").grid(row=row, column=3, columnspan=5, sticky="w")
        row += 1
        
        # ---- Resource-Activity Relationship Section ----
        ttk.Label(increment_frame, text="Resource %:", font=("", 9, "bold")).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.increment_vars['resource'] = tk.StringVar(value="10")
        ttk.Entry(increment_frame, textvariable=self.increment_vars['resource'], width=6).grid(row=row, column=1, sticky="w")
        ttk.Label(increment_frame, text="%/seg", font=("", 8), foreground="gray").grid(row=row, column=2, sticky="w")
        
        # Checkboxes for which activities to apply resource increment
        self.resource_checkboxes = {}
        col = 3
        for activity_key in DEFAULT_ACTIVITIES_CONFIG.keys():
            var = tk.BooleanVar(value=False)  # Default all selected
            self.resource_checkboxes[activity_key] = var
            ttk.Checkbutton(increment_frame, text=activity_key, variable=var).grid(row=row, column=col, sticky="w", padx=2)
            col += 1
        row += 1
        
        # Example preview
        ttk.Label(increment_frame, text="💡 Tip: Uncheck activities you don't want to auto-increment. Resource: Res1↓ Res2↑", 
                  font=("", 8), foreground="blue").grid(row=row, column=0, columnspan=8, sticky="w", pady=5)
        
        # ============ SEGMENTS CONTAINER ============
        self.segments_frame = ttk.Frame(self.scrollable_frame)
        self.segments_frame.pack(fill="x", padx=10, pady=10)
        
        # Add only Before Drift segment initially
        self.add_segment(is_before_drift=True)
        
        # Buttons for adding/removing segments
        btn_frame = ttk.Frame(self.scrollable_frame)
        btn_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(btn_frame, text="+ Add Drift Segment (with auto-increment)", command=self.add_drift_segment).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="- Remove Last Segment", command=self.remove_last_segment).pack(side="left", padx=5)
        
        # Output settings
        output_frame = ttk.LabelFrame(self.scrollable_frame, text="Output Settings", padding=10)
        output_frame.pack(fill="x", padx=10, pady=10)
        
        # Filename
        ttk.Label(output_frame, text="Output Filename:").grid(row=0, column=0, sticky="w", pady=2)
        self.filename_var = tk.StringVar(value="hospital_concept_drift")
        ttk.Entry(output_frame, textvariable=self.filename_var, width=30).grid(row=0, column=1, sticky="w", pady=2)
        
        # Folder
        ttk.Label(output_frame, text="Output Folder:").grid(row=1, column=0, sticky="w", pady=2)
        self.folder_var = tk.StringVar(value="datasets/GeneratedUI")
        folder_entry = ttk.Entry(output_frame, textvariable=self.folder_var, width=30)
        folder_entry.grid(row=1, column=1, sticky="w", pady=2)
        ttk.Button(output_frame, text="Browse", command=self.browse_folder).grid(row=1, column=2, padx=5)
        
        # Export format
        ttk.Label(output_frame, text="Export Format:").grid(row=2, column=0, sticky="w", pady=2)
        self.export_var = tk.StringVar(value="XES + CSV")
        export_combo = ttk.Combobox(output_frame, textvariable=self.export_var, 
                                    values=["XES only", "XES + CSV", "XES + Excel", "XES + CSV + Excel"],
                                    state="readonly", width=20)
        export_combo.grid(row=2, column=1, sticky="w", pady=2)
        
        # Generate button
        generate_btn = ttk.Button(self.scrollable_frame, text="🚀 Generate Event Log", 
                                 command=self.generate_log, style="Accent.TButton")
        generate_btn.pack(pady=20)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self.scrollable_frame, textvariable=self.status_var, font=("", 9))
        status_label.pack(pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.scrollable_frame, length=400, mode='determinate')
        self.progress.pack(pady=5)
    
    def add_segment(self, is_before_drift=False):
        """Add a new segment frame."""
        segment_num = len(self.segments)
        
        # Get previous segment's config if available
        prev_config = None
        if len(self.segments) > 0:
            prev_config = self.segments[-1].get_config()
            
            # Apply auto-increment for drift segments (not for segment 0 and 1)
            if not is_before_drift and segment_num >= 1:
                prev_config = self.apply_auto_increment(prev_config, segment_num)
        
        frame = SegmentFrame(self.segments_frame, segment_num, is_before_drift, prev_config)
        frame.pack(fill="x", pady=5)
        self.segments.append(frame)
    
    def apply_auto_increment(self, config, segment_num):
        """Apply auto-increment values to config for new segment."""
        new_config = copy.deepcopy(config)
        
        try:
            # Duration increment
            duration_inc = float(self.increment_vars['duration'].get() or 0)
            # Duration increment - only for selected activities
            duration_inc = float(self.increment_vars['duration'].get() or 0)
            for activity_key in new_config['config']:
                if self.duration_checkboxes[activity_key].get():  # Only if checkbox is checked
                    new_config['config'][activity_key]['mean_duration'] += duration_inc
            
            # XOR probability increment (convert from percentage)
            xor_inc = float(self.increment_vars['xor_prob'].get() or 0) / 100.0
            new_config['xor_prob'] = min(1.0, max(0.0, new_config['xor_prob'] + xor_inc))
            
            # Case gap increment
            gap_inc = float(self.increment_vars['case_gap'].get() or 0)
            new_config['case_gap'] += gap_inc
            
            # Resource percentage increment - only for selected activities
            # First resource decreases, second resource increases
            res_inc = float(self.increment_vars['resource'].get() or 0) / 100.0
            for activity_key in new_config['config']:
                if self.resource_checkboxes[activity_key].get():  # Only if checkbox is checked
                    resources = list(new_config['config'][activity_key]['resources'].keys())
                    if len(resources) >= 2:
                        # Decrease first resource, increase second
                        res1_val = new_config['config'][activity_key]['resources'][resources[0]]
                        res2_val = new_config['config'][activity_key]['resources'][resources[1]]
                        
                        new_res1 = max(0.0, res1_val - res_inc)
                        new_res2 = min(1.0, res2_val + res_inc)
                        
                        # Normalize to ensure they sum to 1
                        total = new_res1 + new_res2
                        if total > 0:
                            new_config['config'][activity_key]['resources'][resources[0]] = new_res1 / total
                            new_config['config'][activity_key]['resources'][resources[1]] = new_res2 / total
        except (ValueError, KeyError):
            pass  # Keep original values if increment fails
        
        return new_config
    
    def add_drift_segment(self):
        """Add a new drift segment."""
        self.add_segment(is_before_drift=False)
        self.status_var.set(f"Added drift segment #{len(self.segments)-1} with auto-increment applied")
    
    def remove_last_segment(self):
        """Remove the last segment (keep at least 1 - Before Drift)."""
        if len(self.segments) > 1:
            frame = self.segments.pop()
            frame.destroy()
            self.status_var.set(f"Removed last segment. {len(self.segments)} segments remaining.")
        else:
            messagebox.showwarning("Warning", "Must have at least 1 segment (Before Drift)")
    
    def browse_folder(self):
        """Browse for output folder."""
        folder = filedialog.askdirectory()
        if folder:
            self.folder_var.set(folder)
    
    def generate_log(self):
        """Generate the event log."""
        try:
            self.status_var.set("Generating event log...")
            self.progress['value'] = 0
            self.root.update()
            
            # Get all segment configurations
            segment_configs = [seg.get_config() for seg in self.segments]
            
            # Create output folder
            output_folder = self.folder_var.get()
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            # Start time
            start_time = datetime(2025, 1, 1, 8, 0, 0)
            
            # Generate first segment (before drift)
            first_config = segment_configs[0]
            segment_start_case_ids = [1]
            merged_log, last_time, next_case_id = generate_event_log(
                first_config['traces'], start_time, first_config['config'],
                start_case_id=1, xor_probability=first_config['xor_prob'],
                min_gap=first_config['min_gap'], max_gap=first_config['max_gap']
            )
            
            self.progress['value'] = 100 / len(segment_configs)
            self.root.update()
            
            # Generate drift segments
            for i, seg_config in enumerate(segment_configs[1:], 1):
                segment_start_case_ids.append(next_case_id)
                segment_log, last_time, next_case_id = generate_event_log(
                    seg_config['traces'], last_time, seg_config['config'],
                    start_case_id=next_case_id, xor_probability=seg_config['xor_prob'],
                    min_gap=seg_config['min_gap'], max_gap=seg_config['max_gap']
                )
                merged_log = merge_event_logs(merged_log, segment_log)
                
                self.progress['value'] = ((i + 1) / len(segment_configs)) * 100
                self.root.update()
            
            # Export
            base_filename = self.filename_var.get()
            export_format = self.export_var.get()
            
            # XES
            xes_path = os.path.join(output_folder, f"{base_filename}.xes")
            xes_exporter.apply(merged_log, xes_path)
            
            # CSV/Excel
            if "CSV" in export_format or "Excel" in export_format:
                df_raw = log_converter.apply(merged_log, variant=log_converter.Variants.TO_DATA_FRAME)
                
                readable_data = []
                for case_id in df_raw['case:concept:name'].unique():
                    case_events = df_raw[df_raw['case:concept:name'] == case_id].sort_values('time:timestamp')
                    activity_groups = case_events.groupby('concept:name')
                    
                    for activity_name, events in activity_groups:
                        start_event = events[events['lifecycle:transition'] == 'start']
                        complete_event = events[events['lifecycle:transition'] == 'complete']
                        
                        if not start_event.empty and not complete_event.empty:
                            readable_data.append({
                                'Case ID': case_id,
                                'Activity': activity_name,
                                'Resource': start_event.iloc[0]['org:resource'],
                                'Start Timestamp': start_event.iloc[0]['time:timestamp'],
                                'Complete Timestamp': complete_event.iloc[0]['time:timestamp']
                            })
                
                df_readable = pd.DataFrame(readable_data)
                df_readable = df_readable.sort_values('Start Timestamp').reset_index(drop=True)
                
                if "CSV" in export_format:
                    csv_path = os.path.join(output_folder, f"{base_filename}.csv")
                    df_readable.to_csv(csv_path, index=False)
                
                if "Excel" in export_format:
                    excel_path = os.path.join(output_folder, f"{base_filename}.xlsx")
                    df_readable.to_excel(excel_path, index=False)
            
            # Generate summary
            summary_lines = self.generate_summary(segment_configs, segment_start_case_ids, base_filename, export_format)
            summary_path = os.path.join(output_folder, f"{base_filename}_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(summary_lines))
            
            total_traces = sum(seg['traces'] for seg in segment_configs)
            self.progress['value'] = 100
            self.status_var.set(f"✅ Generated {total_traces} traces! Saved to {output_folder}/")
            
            messagebox.showinfo("Success", 
                f"Event log generated successfully!\n\n"
                f"Total traces: {total_traces}\n"
                f"Location: {output_folder}/\n"
                f"Files: {base_filename}.xes" + 
                (f", {base_filename}.csv" if "CSV" in export_format else "") +
                (f", {base_filename}.xlsx" if "Excel" in export_format else ""))
            
        except Exception as e:
            self.status_var.set(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate event log:\n{str(e)}")
    
    def generate_summary(self, segment_configs, segment_start_case_ids, base_filename, export_format) -> list:
        """Generate summary text showing only what changed and at which case ID."""
        lines = []
        lines.append("=" * 60)
        lines.append("EVENT LOG GENERATION SUMMARY")
        lines.append("=" * 60)
        lines.append(f"\nGeneration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total traces: {sum(seg['traces'] for seg in segment_configs)}")
        lines.append(f"Segments: {len(segment_configs)} (1 baseline + {len(segment_configs)-1} drift)")

        # Baseline info
        base = segment_configs[0]
        lines.append(f"\n{'-' * 60}")
        lines.append("BASELINE (Segment 0) - Case IDs 1 to {}:".format(
            segment_start_case_ids[1] - 1 if len(segment_start_case_ids) > 1 else base['traces']
        ))
        lines.append(f"{'-' * 60}")
        lines.append(f"  Traces        : {base['traces']}")
        lines.append(f"  Lab Test prob : {base['xor_prob']:.0%}")
        lines.append(f"  Case Gap      : {base['case_gap']} min")
        lines.append(f"  Activity durations:")
        for key, data in base['config'].items():
            lines.append(f"    {key} ({data['name']}): {data['mean_duration']} min")
        lines.append(f"  Resource distributions:")
        for key, data in base['config'].items():
            for resource, prob in data['resources'].items():
                lines.append(f"    {key} - {resource}: {prob*100:.1f}%")

        # Drift segments - only show changes
        lines.append(f"\n{'-' * 60}")
        lines.append("DRIFT CHANGES")
        lines.append(f"{'-' * 60}")

        for i in range(1, len(segment_configs)):
            prev = segment_configs[i - 1]
            curr = segment_configs[i]
            start_case = segment_start_case_ids[i]
            end_case = segment_start_case_ids[i + 1] - 1 if i + 1 < len(segment_start_case_ids) else start_case + curr['traces'] - 1

            changes = []

            # XOR probability
            if abs(curr['xor_prob'] - prev['xor_prob']) > 0.001:
                changes.append(f"  At case {start_case}: Lab Test probability changed from {prev['xor_prob']:.0%} to {curr['xor_prob']:.0%}")

            # Case gap
            if abs(curr['case_gap'] - prev['case_gap']) > 0.01:
                changes.append(f"  At case {start_case}: Case gap changed from {prev['case_gap']} min to {curr['case_gap']} min")

            # Activity durations
            for key in curr['config']:
                curr_dur = curr['config'][key]['mean_duration']
                prev_dur = prev['config'][key]['mean_duration']
                if abs(curr_dur - prev_dur) > 0.01:
                    name = curr['config'][key]['name']
                    changes.append(f"  At case {start_case}: Activity {key} ({name}) duration changed from {prev_dur} min to {curr_dur} min")

            # Resource probabilities
            for key in curr['config']:
                for resource in curr['config'][key]['resources']:
                    curr_prob = curr['config'][key]['resources'][resource]
                    prev_prob = prev['config'][key]['resources'][resource]
                    if abs(curr_prob - prev_prob) > 0.001:
                        changes.append(f"  At case {start_case}: {key} - {resource} changed from {prev_prob*100:.1f}% to {curr_prob*100:.1f}%")

            if changes:
                lines.append(f"\nSegment {i} (Case IDs {start_case}-{end_case}):")
                lines.extend(changes)
            else:
                lines.append(f"\nSegment {i} (Case IDs {start_case}-{end_case}): No changes")

        lines.append(f"\n{'-' * 60}")
        lines.append("OUTPUT FILES")
        lines.append(f"{'-' * 60}")
        lines.append(f"XES file: {base_filename}.xes")
        if "CSV" in export_format:
            lines.append(f"CSV file: {base_filename}.csv")
        if "Excel" in export_format:
            lines.append(f"Excel file: {base_filename}.xlsx")
        lines.append(f"Summary file: {base_filename}_summary.txt")
        lines.append("\n" + "=" * 60)

        return lines


def main():
    root = tk.Tk()
    
    # Style
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except:
        pass
    
    app = EventLogGeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
