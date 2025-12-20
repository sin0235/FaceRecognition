"""
Database Builder Module
Tạo database embeddings cho ArcFace, FaceNet và LBPH model
Hỗ trợ job tracking và progress monitoring
"""

import os
import sys
import threading
import time
import traceback
from typing import Dict, Optional, Callable
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


class BuildJob:
    """Class theo dõi trạng thái của một job build database"""
    
    def __init__(self, job_id: str, model_type: str, config: Dict):
        self.job_id = job_id
        self.model_type = model_type
        self.config = config
        self.status = "pending"  # pending, running, completed, failed
        self.progress = 0.0
        self.message = "Đang khởi tạo..."
        self.logs = []
        self.output_files = {}
        self.error = None
        self.start_time = None
        self.end_time = None
    
    def add_log(self, message: str):
        """Thêm log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)
    
    def update_progress(self, progress: float, message: str = None):
        """Cập nhật progress và message"""
        self.progress = min(100.0, max(0.0, progress))
        if message:
            self.message = message
            self.add_log(message)
    
    def set_status(self, status: str):
        """Set trạng thái job"""
        self.status = status
        if status == "running":
            self.start_time = datetime.now()
        elif status in ["completed", "failed"]:
            self.end_time = datetime.now()
    
    def set_error(self, error: str):
        """Set error message"""
        self.error = error
        self.add_log(f"ERROR: {error}")
    
    def add_output_file(self, label: str, path: str):
        """Thêm file output"""
        self.output_files[label] = path
        self.add_log(f"Created: {label} -> {path}")
    
    def to_dict(self) -> Dict:
        """Chuyển đổi sang dict để trả về API"""
        return {
            "job_id": self.job_id,
            "model_type": self.model_type,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "logs": self.logs[-50:],  # Chỉ trả 50 log gần nhất
            "output_files": self.output_files,
            "error": self.error,
            "elapsed_time": self._get_elapsed_time()
        }
    
    def _get_elapsed_time(self) -> Optional[float]:
        """Tính thời gian đã chạy"""
        if not self.start_time:
            return None
        end = self.end_time if self.end_time else datetime.now()
        return (end - self.start_time).total_seconds()


class DatabaseBuilder:
    """Quản lý việc build database cho các model"""
    
    def __init__(self):
        self.jobs: Dict[str, BuildJob] = {}
        self.lock = threading.Lock()
    
    def create_job(self, job_id: str, model_type: str, config: Dict) -> BuildJob:
        """Tạo job mới"""
        with self.lock:
            job = BuildJob(job_id, model_type, config)
            self.jobs[job_id] = job
            return job
    
    def get_job(self, job_id: str) -> Optional[BuildJob]:
        """Lấy job theo ID"""
        with self.lock:
            return self.jobs.get(job_id)
    
    def start_build(self, job_id: str):
        """Bắt đầu build database trong background thread"""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} không tồn tại")
        
        thread = threading.Thread(target=self._run_build, args=(job,), daemon=True)
        thread.start()
    
    def _run_build(self, job: BuildJob):
        """Chạy build process (được gọi trong thread riêng)"""
        try:
            job.set_status("running")
            job.update_progress(5, "Đang khởi tạo build process...")
            
            if job.model_type == "lbph":
                self._build_lbph(job)
            elif job.model_type == "arcface":
                self._build_arcface(job)
            elif job.model_type == "facenet":
                self._build_facenet(job)
            else:
                raise ValueError(f"Model type không hợp lệ: {job.model_type}")
            
            job.update_progress(100, "Hoàn thành!")
            job.set_status("completed")
            
        except Exception as e:
            job.set_error(str(e))
            job.add_log(traceback.format_exc())
            job.set_status("failed")
    
    def _build_lbph(self, job: BuildJob):
        """Build LBPH model từ dataset"""
        from models.lbphmodel.train_lbph_script import train_lbph_from_directory
        
        job.update_progress(10, "Đang load LBPH config...")
        
        config = job.config
        data_dir = config.get("data_dir")
        output_dir = config.get("output_dir", "models/checkpoints/LBHP")
        model_name = config.get("model_name", "lbph_model.xml")
        
        # LBPH parameters
        radius = config.get("radius", 1)
        neighbors = config.get("neighbors", 8)
        grid_x = config.get("grid_x", 8)
        grid_y = config.get("grid_y", 8)
        use_face_detection = config.get("use_face_detection", True)
        target_size = tuple(config.get("target_size", [100, 100]))
        
        # Threshold finding
        find_threshold = config.get("find_threshold", False)
        val_dir = config.get("val_dir")
        
        job.update_progress(20, f"Đang train LBPH từ {data_dir}...")
        
        model, label_map = train_lbph_from_directory(
            data_dir=data_dir,
            output_dir=output_dir,
            model_name=model_name,
            radius=radius,
            neighbors=neighbors,
            grid_x=grid_x,
            grid_y=grid_y,
            use_val_for_threshold=find_threshold,
            val_dir=val_dir,
            use_face_detection=use_face_detection,
            target_size=target_size
        )
        
        model_path = os.path.join(output_dir, model_name)
        
        # Chỉ add model file chính, không add label_map và threshold
        job.add_output_file("LBPH Model", model_path)
    
    def _build_arcface(self, job: BuildJob):
        """Build ArcFace embeddings database"""
        from inference.extract_embeddings import build_db
        
        job.update_progress(10, "Đang load ArcFace model...")
        
        config = job.config
        model_path = config.get("model_path")
        root_folder = config.get("data_dir")
        save_path = config.get("output_path", "data/arcface_embeddings_db.npy")
        device = config.get("device", "cpu")
        use_face_detection = config.get("use_face_detection", True)
        
        job.update_progress(20, f"Đang extract embeddings từ {root_folder}...")
        
        build_db(
            model_path=model_path,
            root_folder=root_folder,
            save_path=save_path,
            device=device,
            use_face_detection=use_face_detection,
            model_type="arcface"
        )
        
        job.add_output_file("ArcFace Database", save_path)
    
    def _build_facenet(self, job: BuildJob):
        """Build FaceNet embeddings database"""
        from inference.extract_embeddings import build_db
        
        job.update_progress(10, "Đang load FaceNet model...")
        
        config = job.config
        model_path = config.get("model_path")
        root_folder = config.get("data_dir")
        save_path = config.get("output_path", "data/facenet_embeddings_db.npy")
        device = config.get("device", "cpu")
        use_face_detection = config.get("use_face_detection", True)
        
        job.update_progress(20, f"Đang extract embeddings từ {root_folder}...")
        
        build_db(
            model_path=model_path,
            root_folder=root_folder,
            save_path=save_path,
            device=device,
            use_face_detection=use_face_detection,
            model_type="facenet"
        )
        
        job.add_output_file("FaceNet Database", save_path)


# Global builder instance
_builder = DatabaseBuilder()


def get_builder() -> DatabaseBuilder:
    """Lấy global builder instance"""
    return _builder
