"""
Time estimation utilities for document processing
"""
import time
from typing import Optional


class ProcessingTimer:
    """Tracks processing time and estimates remaining time with smart extrapolation"""
    
    def __init__(self, total_items: int):
        self.total_items = total_items
        self.processed_items = 0
        self.start_time = time.time()
        self.item_times = []  # Track individual item processing times
        self.last_item_start = None
    
    def start_item(self):
        """Mark the start of processing an item"""
        self.last_item_start = time.time()
    
    def complete_item(self):
        """Mark completion of an item and record its processing time"""
        if self.last_item_start is not None:
            item_duration = time.time() - self.last_item_start
            self.item_times.append(item_duration)
            self.processed_items += 1
            self.last_item_start = None
    
    def update(self, items_processed: int = 1):
        """Update progress (legacy method, kept for compatibility)"""
        self.processed_items += items_processed
    
    def get_estimated_remaining(self) -> Optional[float]:
        """
        Get estimated remaining time in seconds using smart extrapolation:
        - First item: Return None (still calculating)
        - After first item: Use first item's time for all remaining
        - After second+ items: Use weighted average with more weight on recent items
        """
        if self.processed_items == 0:
            return None  # Still processing first item
        
        remaining_items = self.total_items - self.processed_items
        
        if remaining_items <= 0:
            return 0
        
        if len(self.item_times) == 0:
            # Fallback to simple elapsed time calculation
            elapsed = time.time() - self.start_time
            avg_time_per_item = elapsed / self.processed_items
            return avg_time_per_item * remaining_items
        
        elif len(self.item_times) == 1:
            # After first item: assume same time for all remaining
            return self.item_times[0] * remaining_items
        
        else:
            # After 2+ items: use weighted average (more weight on recent items)
            # This accounts for variations in document sizes
            weights = [i + 1 for i in range(len(self.item_times))]  # 1, 2, 3, ...
            weighted_sum = sum(t * w for t, w in zip(self.item_times, weights))
            total_weight = sum(weights)
            avg_time_per_item = weighted_sum / total_weight
            
            return avg_time_per_item * remaining_items
    
    def get_progress_percentage(self) -> float:
        """Get progress as percentage"""
        if self.total_items == 0:
            return 100.0
        return (self.processed_items / self.total_items) * 100
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time
    
    @staticmethod
    def format_time(seconds: Optional[float], show_calculating: bool = True) -> str:
        """Format seconds into human-readable string"""
        if seconds is None:
            if show_calculating:
                return "Calculating, please be patient..."
            else:
                return "Calculating..."
        
        if seconds < 0:
            return "Almost done..."
        
        if seconds < 1:
            return "Less than 1s"
        elif seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
