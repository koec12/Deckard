"""Modbus TCP server for exposing ROI detection status and object information."""

import asyncio
from pymodbus.server import StartTcpServer, ModbusTcpServer
from pymodbus import ModbusDeviceIdentification
from pymodbus.datastore import ModbusDeviceContext, ModbusServerContext
from pymodbus.datastore import ModbusSequentialDataBlock
import threading
from typing import Dict, List, Optional
from tracker import TrackedObject


class ModbusDataStore:
    """Custom data store for Modbus registers."""
    
    def __init__(self, max_rois=9, max_objects_per_roi=99):
        self.max_rois = max_rois
        self.max_objects_per_roi = max_objects_per_roi
        self._last_object_counts = {roi_id: 0 for roi_id in range(1, max_rois + 1)}
        
        # ROI status registers: 11-19 (need up to index 19)
        # Object registers: 10 registers per object
        # ROI 1 objects: 1000-1999 (100 objects * 10 = 1000 registers)
        # ROI 2 objects: 2000-2999, etc.
        # ROI 9, object 99: 9990-9999 (need up to index 9999)
        total_registers = 10000
        
        # Initialize holding registers (all zeros)
        self.holding_registers = ModbusSequentialDataBlock(0, [0] * total_registers)
        
        # Create device context
        self.device_context = ModbusDeviceContext(
            di=ModbusSequentialDataBlock(0, [0] * 100),  # Discrete inputs
            co=ModbusSequentialDataBlock(0, [0] * 100),  # Coils
            hr=self.holding_registers,  # Holding registers
            ir=ModbusSequentialDataBlock(0, [0] * 100)   # Input registers
        )
        
        # Create server context
        self.server_context = ModbusServerContext(devices=self.device_context, single=True)
    
    def update_roi_status(self, roi_id: int, objects: List[TrackedObject]):
        """
        Update ROI status register.
        
        Args:
            roi_id: ROI ID (1-9)
            objects: List of tracked objects in the ROI
        """
        if roi_id < 1 or roi_id > self.max_rois:
            return
        
        # Register address: 11 + (roi_id - 1)
        register_addr = 10 + roi_id
        
        # Initialize status bits
        status = 0
        
        # Check for each class
        for obj in objects:
            class_id = obj.current_class_id
            if class_id == 1:  # Red
                status |= (1 << 0)
            elif class_id == 2:  # Silver/Grey
                status |= (1 << 1)
            elif class_id == 0:  # Black
                status |= (1 << 2)
        
        # Update register
        self.holding_registers.setValues(register_addr, [status])
    
    def update_object_registers(self, roi_id: int, objects: List[TrackedObject], pixels_per_cm: float, fps: float = 30.0):
        """
        Update object information registers for a ROI.
        
        Args:
            roi_id: ROI ID (1-9)
            objects: List of tracked objects in the ROI
            pixels_per_cm: Pixels per cm ratio for speed calculation
            fps: Current frames per second
        """
        if roi_id < 1 or roi_id > self.max_rois:
            return
        
        # Limit to max objects per ROI
        objects = objects[:self.max_objects_per_roi]
        
        # Base address for ROI: roi_id * 1000
        base_addr = roi_id * 1000
        
        # Update registers for each object
        for idx, obj in enumerate(objects):
            # Object registers start at base_addr + (idx * 10)
            obj_base = base_addr + (idx * 10)
            
            # Use cached speed when available
            speed = getattr(obj, "cached_speed", None)
            if speed is None:
                speed = obj.calculate_speed(pixels_per_cm, fps)
                obj.cached_speed = speed
            
            # Prepare register values
            # xyy0: Object ID
            # xyy1: Class ID
            # xyy2: Confidence (0-100%)
            # xyy3: Speed (cm/s) - convert to integer (multiply by 10 for 0.1 cm/s precision)
            # xyy4: Centroid X
            # xyy5: Centroid Y
            # xyy6-xyy9: Reserved (0)
            
            confidence_percent = int(obj.current_confidence * 100)
            speed_int = int(speed * 10)  # 0.1 cm/s precision
            
            values = [
                obj.object_id,           # xyy0
                obj.current_class_id,    # xyy1
                confidence_percent,      # xyy2
                speed_int,               # xyy3
                obj.centroid[0],         # xyy4 (centroid x)
                obj.centroid[1],         # xyy5 (centroid y)
                0, 0, 0, 0         # xyy6-xyy9 (reserved)
            ]
            
            self.holding_registers.setValues(obj_base, values)
        
        # Clear registers only for slots that were previously used
        current_count = len(objects)
        last_count = self._last_object_counts.get(roi_id, 0)
        if last_count > current_count:
            for idx in range(current_count, last_count):
                obj_base = base_addr + (idx * 10)
                values = [0] * 10
                self.holding_registers.setValues(obj_base, values)
        self._last_object_counts[roi_id] = current_count
    
    def update_all_rois(self, roi_data: Dict[int, List[TrackedObject]], pixels_per_cm: float, fps: float = 30.0):
        """
        Update all ROI registers.
        
        Args:
            roi_data: Dictionary mapping ROI ID to list of tracked objects
            pixels_per_cm: Pixels per cm ratio for speed calculation
            fps: Current frames per second (default 30.0)
        """
        for roi_id in range(1, self.max_rois + 1):
            objects = roi_data.get(roi_id, [])
            self.update_roi_status(roi_id, objects)
            self.update_object_registers(roi_id, objects, pixels_per_cm, fps)


class ModbusServer:
    """Modbus TCP Server wrapper."""
    
    def __init__(self, host="0.0.0.0", port=502, max_rois=9, max_objects_per_roi=99):
        self.host = host
        self.port = port
        self.data_store = ModbusDataStore(max_rois, max_objects_per_roi)
        self.server_thread = None
        self.running = False
        self._server = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Device identification
        self.identity = ModbusDeviceIdentification()
        self.identity.VendorName = 'Deckard Cylinder Tracking System'
        self.identity.ProductCode = 'DCTS'
        self.identity.VendorUrl = ''
        self.identity.ProductName = 'Deckard Cylinder Tracker'
        self.identity.ModelName = 'DCTS-1.0'
        self.identity.MajorMinorRevision = '1.0.0'
    
    def start(self):
        """Start the Modbus TCP server in a separate thread."""
        if self.running:
            return
        
        def run_server():
            try:
                if ModbusTcpServer is not None:
                    # pymodbus 3.x servers are asyncio-based; create a loop in this thread.
                    self._loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._loop)

                    async def _serve() -> None:
                        # Important: create the server while the loop is running.
                        self._server = ModbusTcpServer(
                            context=self.data_store.server_context,
                            identity=self.identity,
                            address=(self.host, self.port),
                        )
                        await self._server.serve_forever()

                    # Blocks until shutdown is requested.
                    self._loop.run_until_complete(_serve())
                else:
                    # Very old pymodbus fallback (blocking, no clean shutdown).
                    StartTcpServer(
                        context=self.data_store.server_context,
                        identity=self.identity,
                        address=(self.host, self.port),
                    )
            except Exception as e:
                print(f"Modbus server error: {e}")
            finally:
                loop = self._loop
                self._loop = None
                if loop is not None:
                    try:
                        loop.close()
                    except Exception:
                        pass
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.running = True
        print(f"Modbus TCP server started on {self.host}:{self.port}")
    
    def stop(self):
        """Stop the Modbus TCP server."""
        server = self._server
        loop = self._loop

        # Clean shutdown for pymodbus 3.x (asyncio)
        if server is not None and loop is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(server.shutdown(), loop)
                fut.result(timeout=5)
            except Exception:
                pass

        self._server = None

        # Note: StartTcpServer fallback doesn't have a clean stop method
        self.running = False

        if self.server_thread is not None:
            try:
                self.server_thread.join(timeout=5)
            except Exception:
                pass
            self.server_thread = None
    
    def update_registers(self, roi_data: Dict[int, List[TrackedObject]], pixels_per_cm: float, fps: float = 30.0):
        """Update all Modbus registers with current detection data."""
        if self.running:
            self.data_store.update_all_rois(roi_data, pixels_per_cm, fps)

