#!/usr/bin/env python3
"""
Hardware Acceleration Utilities
===============================

Provides hardware acceleration support for UAP analysis across different platforms:
- macOS: Metal Performance Shaders (MPS) 
- Linux/Windows: CUDA
- Fallback: CPU with optimizations
"""

import platform
import logging
import numpy as np

logger = logging.getLogger(__name__)

class AccelerationManager:
    """Manages hardware acceleration across different platforms."""
    
    def __init__(self, config=None):
        """Initialize acceleration manager."""
        self.config = config or {}
        self.device_type = "cpu"
        self.device = None
        self.backend = None
        
        # Detect and initialize best available acceleration
        self._detect_acceleration()
    
    def _detect_acceleration(self):
        """Detect and initialize the best available acceleration."""
        system = platform.system()
        
        # Try to initialize acceleration in order of preference
        if system == "Darwin":  # macOS
            if self._init_metal_mps():
                return
        
        # Try CUDA (Linux/Windows)
        if self._init_cuda():
            return
            
        # Try OpenCL (cross-platform)
        if self._init_opencl():
            return
        
        # Fallback to optimized CPU
        self._init_cpu_optimized()
    
    def _init_metal_mps(self):
        """Initialize Metal Performance Shaders on macOS."""
        try:
            import torch
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.device_type = "mps"
                self.backend = "metal"
                logger.info("✅ Metal Performance Shaders (MPS) acceleration enabled")
                return True
        except ImportError:
            logger.info("PyTorch not available for MPS acceleration")
        except Exception as e:
            logger.warning(f"MPS initialization failed: {e}")
        
        return False
    
    def _init_cuda(self):
        """Initialize CUDA acceleration."""
        try:
            import torch
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.device_type = "cuda"
                self.backend = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✅ CUDA acceleration enabled on {gpu_name}")
                return True
        except ImportError:
            logger.info("PyTorch not available for CUDA acceleration")
        except Exception as e:
            logger.warning(f"CUDA initialization failed: {e}")
        
        return False
    
    def _init_opencl(self):
        """Initialize OpenCL acceleration."""
        try:
            import cv2
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                self.device_type = "opencl"
                self.backend = "opencv_opencl"
                logger.info("✅ OpenCL acceleration enabled")
                return True
        except Exception as e:
            logger.warning(f"OpenCL initialization failed: {e}")
        
        return False
    
    def _init_cpu_optimized(self):
        """Initialize optimized CPU processing."""
        try:
            import cv2
            # Enable optimizations
            cv2.setUseOptimized(True)
            cv2.setNumThreads(0)  # Use all available cores
            
            self.device_type = "cpu"
            self.backend = "cpu_optimized"
            logger.info("✅ Optimized CPU processing enabled")
        except Exception as e:
            logger.warning(f"CPU optimization failed: {e}")
            self.backend = "cpu_basic"
    
    def get_device_info(self):
        """Get information about the current acceleration device."""
        info = {
            "device_type": self.device_type,
            "backend": self.backend,
            "platform": platform.system()
        }
        
        if self.device_type == "cuda":
            try:
                import torch
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
                info["cuda_version"] = torch.version.cuda
            except:
                pass
        
        elif self.device_type == "mps":
            try:
                import torch
                info["pytorch_version"] = torch.__version__
                info["mps_available"] = torch.backends.mps.is_available()
            except:
                pass
        
        return info
    
    def accelerate_optical_flow(self, prev_frame, curr_frame, **params):
        """Accelerated optical flow computation."""
        if self.device_type in ["cuda", "mps"]:
            return self._torch_optical_flow(prev_frame, curr_frame, **params)
        elif self.device_type == "opencl":
            return self._opencv_opencl_flow(prev_frame, curr_frame, **params)
        else:
            return self._cpu_optical_flow(prev_frame, curr_frame, **params)
    
    def _torch_optical_flow(self, prev_frame, curr_frame, **params):
        """PyTorch-based optical flow (MPS/CUDA)."""
        try:
            import torch
            import cv2
            
            # Convert to torch tensors
            prev_tensor = torch.from_numpy(prev_frame).float().to(self.device)
            curr_tensor = torch.from_numpy(curr_frame).float().to(self.device)
            
            # Use OpenCV for now, but move data to GPU
            prev_cpu = prev_tensor.cpu().numpy().astype(np.uint8)
            curr_cpu = curr_tensor.cpu().numpy().astype(np.uint8)
            
            # Compute flow on CPU (OpenCV doesn't have GPU Farneback yet)
            flow = cv2.calcOpticalFlowFarneback(
                prev_cpu, curr_cpu, None,
                params.get('pyr_scale', 0.5),
                params.get('levels', 3),
                params.get('winsize', 15),
                params.get('iterations', 3),
                params.get('poly_n', 5),
                params.get('poly_sigma', 1.2),
                0
            )
            
            return flow
            
        except Exception as e:
            logger.warning(f"Torch optical flow failed: {e}, falling back to CPU")
            return self._cpu_optical_flow(prev_frame, curr_frame, **params)
    
    def _opencv_opencl_flow(self, prev_frame, curr_frame, **params):
        """OpenCL-accelerated optical flow."""
        try:
            import cv2
            
            # Upload to GPU
            prev_gpu = cv2.UMat(prev_frame)
            curr_gpu = cv2.UMat(curr_frame)
            
            # Compute flow on GPU
            flow = cv2.calcOpticalFlowFarneback(
                prev_gpu, curr_gpu, None,
                params.get('pyr_scale', 0.5),
                params.get('levels', 3),
                params.get('winsize', 15),
                params.get('iterations', 3),
                params.get('poly_n', 5),
                params.get('poly_sigma', 1.2),
                0
            )
            
            # Download result
            return flow.get()
            
        except Exception as e:
            logger.warning(f"OpenCL optical flow failed: {e}, falling back to CPU")
            return self._cpu_optical_flow(prev_frame, curr_frame, **params)
    
    def _cpu_optical_flow(self, prev_frame, curr_frame, **params):
        """CPU optical flow computation."""
        import cv2
        
        return cv2.calcOpticalFlowFarneback(
            prev_frame, curr_frame, None,
            params.get('pyr_scale', 0.5),
            params.get('levels', 3),
            params.get('winsize', 15),
            params.get('iterations', 3),
            params.get('poly_n', 5),
            params.get('poly_sigma', 1.2),
            0
        )
    
    def accelerate_filter_operations(self, data, filter_func, **kwargs):
        """Accelerated filtering operations."""
        if self.device_type in ["cuda", "mps"]:
            return self._torch_filter(data, filter_func, **kwargs)
        else:
            return self._numpy_filter(data, filter_func, **kwargs)
    
    def _torch_filter(self, data, filter_func, **kwargs):
        """PyTorch-accelerated filtering."""
        try:
            import torch
            
            # Convert to tensor
            tensor_data = torch.from_numpy(data).float().to(self.device)
            
            # Apply filter (implement common filters)
            if filter_func.__name__ == 'gaussian_filter':
                # Implement Gaussian filter in PyTorch
                result = self._torch_gaussian_filter(tensor_data, **kwargs)
            else:
                # Fallback to CPU
                result = filter_func(data, **kwargs)
                return result
            
            return result.cpu().numpy()
            
        except Exception as e:
            logger.warning(f"Torch filtering failed: {e}, falling back to NumPy")
            return self._numpy_filter(data, filter_func, **kwargs)
    
    def _torch_gaussian_filter(self, tensor_data, sigma=1.0, **kwargs):
        """PyTorch Gaussian filter implementation."""
        import torch
        import torch.nn.functional as F
        
        # Create Gaussian kernel
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 1D Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Reshape for convolution
        if len(tensor_data.shape) == 2:
            # 2D data
            tensor_data = tensor_data.unsqueeze(0).unsqueeze(0)
            kernel_x = kernel_1d.view(1, 1, 1, -1)
            kernel_y = kernel_1d.view(1, 1, -1, 1)
            
            # Apply separable convolution
            result = F.conv2d(tensor_data, kernel_x, padding=(0, kernel_size//2))
            result = F.conv2d(result, kernel_y, padding=(kernel_size//2, 0))
            
            return result.squeeze()
        else:
            # 1D data
            tensor_data = tensor_data.unsqueeze(0).unsqueeze(0)
            kernel_1d = kernel_1d.view(1, 1, -1)
            result = F.conv1d(tensor_data, kernel_1d, padding=kernel_size//2)
            return result.squeeze()
    
    def _numpy_filter(self, data, filter_func, **kwargs):
        """NumPy-based filtering (CPU fallback)."""
        return filter_func(data, **kwargs)
    
    def accelerate_fft(self, data):
        """Accelerated FFT computation."""
        if self.device_type in ["cuda", "mps"]:
            return self._torch_fft(data)
        else:
            return self._numpy_fft(data)
    
    def _torch_fft(self, data):
        """PyTorch FFT."""
        try:
            import torch
            
            tensor_data = torch.from_numpy(data).to(self.device)
            result = torch.fft.fft(tensor_data)
            return result.cpu().numpy()
            
        except Exception as e:
            logger.warning(f"Torch FFT failed: {e}, falling back to NumPy")
            return self._numpy_fft(data)
    
    def _numpy_fft(self, data):
        """NumPy FFT (CPU)."""
        return np.fft.fft(data)

# Global acceleration manager instance
_acceleration_manager = None

def get_acceleration_manager(config=None):
    """Get the global acceleration manager instance."""
    global _acceleration_manager
    if _acceleration_manager is None:
        _acceleration_manager = AccelerationManager(config)
    return _acceleration_manager

def benchmark_acceleration():
    """Benchmark different acceleration methods."""
    import time
    
    print("🚀 Hardware Acceleration Benchmark")
    print("=" * 50)
    
    # Get acceleration manager
    accel = get_acceleration_manager()
    
    # Print device info
    info = accel.get_device_info()
    print(f"Platform: {info['platform']}")
    print(f"Device: {info['device_type']}")
    print(f"Backend: {info['backend']}")
    
    if info['device_type'] == 'cuda':
        print(f"GPU: {info.get('gpu_name', 'Unknown')}")
    
    print()
    
    # Create test data
    test_size = (1000, 1000)
    prev_frame = np.random.randint(0, 255, test_size, dtype=np.uint8)
    curr_frame = np.random.randint(0, 255, test_size, dtype=np.uint8)
    test_data = np.random.randn(10000).astype(np.float32)
    
    # Benchmark optical flow
    print("Testing optical flow...")
    start_time = time.time()
    for _ in range(3):
        flow = accel.accelerate_optical_flow(prev_frame, curr_frame)
    optical_flow_time = (time.time() - start_time) / 3
    print(f"  Average time: {optical_flow_time:.3f}s")
    
    # Benchmark FFT
    print("Testing FFT...")
    start_time = time.time()
    for _ in range(10):
        fft_result = accel.accelerate_fft(test_data)
    fft_time = (time.time() - start_time) / 10
    print(f"  Average time: {fft_time:.4f}s")
    
    print("\n✅ Benchmark complete!")

if __name__ == "__main__":
    benchmark_acceleration()